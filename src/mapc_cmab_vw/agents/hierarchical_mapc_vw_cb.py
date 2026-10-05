from collections import defaultdict
from itertools import chain
from typing import Callable

import numpy as np
from chex import Array

from mapc_cmab_vw.agents.mapc_agent import MapcAgent
from mapc_cmab_vw.agents.vw_contextual_bandit import VwCBAgent


class HierarchicalMapcVWAgent(MapcAgent):
    """
    Hierarchical contextual-bandit agent (Vowpal Wabbit backend) responsible for the
    selection of the AP / station / link / tx-power configuration.

    Identical in structure to ``HierarchicalMapcDQNAgent``:

      1. Each non-sharing AP decides whether it joins the group (binary action).
      2. Each AP in the group selects the (relative) index of the station it serves.
      3. Each AP in the group selects a link combination (index into link_comb_index_to_links).
      4. Each (station, link) pair selects a transmission power index.

    INDEXING CONVENTION
    -------------------
    * Everything is 0-based, exactly as in the DQN agent.
    * ``VwCBAgent`` already exposes 0-based actions; the 0-based <-> VW 1-based
      conversion is isolated inside the wrapper (applied only when the VW learn
      example is built). This class therefore performs NO index shifting.
    * ``_sample`` only validates that the returned action is within 0..K-1, so a
      convention mismatch fails immediately instead of silently wrapping around
      (e.g. -1 indexing the last element).
    * The wrapper stores its own previous action, so the ``*_last_action`` dicts are
      not passed to the agents; they are still maintained (0-based) so logging and
      inspection match the DQN agent.

    Reward bookkeeping is unchanged: ``self.rewards[last_step[agent]]`` is the (1-step
    delayed) reward belonging to the action the agent took when it was last sampled.
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            find_groups_agent: dict[int, VwCBAgent],
            assign_stations_agent: dict[int, VwCBAgent],
            assign_links_agent: dict[int, VwCBAgent],
            assign_tx_power_agent: dict[tuple[int, int], VwCBAgent],
            encode_sharing_ap: Callable,
            encode_ap_group: Callable,
            encode_ap_stations_to_tx_vector: Callable,
            encode_sta_links_vector: Callable,
            ap_group_action_to_ap_group: Callable,
            link_comb_index_to_links: dict[int, list],
            sta_index_mapping: dict[int, int],
            n_links: int,
            n_tx_power_levels: int,
            logger=None
        ):

        self.associations = associations
        self.inv_associations = {sta: ap for ap in associations.keys() for sta in associations[ap]}
        self.find_groups_agent = find_groups_agent
        self.assign_stations_agent = assign_stations_agent
        self.assign_links_agent = assign_links_agent
        self.assign_tx_power_agent = assign_tx_power_agent
        self.ap_group_action_to_ap_group = ap_group_action_to_ap_group
        self.link_comb_index_to_links = link_comb_index_to_links
        self.sta_index_mapping = sta_index_mapping
        self.n_links = n_links
        self.logger = logger

        self.encoded_sharing_ap = encode_sharing_ap
        self.encode_ap_group = encode_ap_group
        self.encode_ap_stations_to_tx_vector = encode_ap_stations_to_tx_vector
        self.encode_sta_links_vector = encode_sta_links_vector

        self.n_tx_power_levels = n_tx_power_levels

        # step at which an agent last acted (-> index of its delayed reward)
        # last action is kept 0-based, same as in the DQN agent (bookkeeping only)
        self.find_groups_agent_last_step = defaultdict(int)
        self.find_groups_agent_last_action = defaultdict(int)

        self.assign_stations_agent_last_step = defaultdict(int)
        self.assign_stations_agent_last_action = defaultdict(int)

        self.assign_links_agent_last_step = defaultdict(int)
        self.assign_links_agent_last_action = defaultdict(int)

        self.assign_tx_power_agent_last_step = defaultdict(int)
        self.assign_tx_power_agent_last_action = defaultdict(int)

        self.step = 0
        self.rewards = []

        self.associations = {ap: np.array(stations) for ap, stations in associations.items()}
        self.access_points = np.asarray(list(associations.keys()))
        self.stations = np.asarray(list(chain.from_iterable(associations.values())))
        self.n_nodes = len(self.access_points) + len(list(chain.from_iterable(associations.values())))
        self.n_ap = len(associations.keys())

    # ------------------------------------------------------------------ #
    # Single point of contact with the VW wrapper
    # ------------------------------------------------------------------ #
    def _sample(self, agent: VwCBAgent, context: Array, last_step: int) -> int:
        """
        Learns from the delayed reward of the agent's previous action and returns the
        new action as a 0-based index.

        ``VwCBAgent.sample(context, previous_reward)`` internally
          1. learns on (previous_context, previous_action, previous_reward),
          2. predicts for the new context and returns a 0-based action.
        """
        previous_reward = float(self.rewards[last_step])
        action = int(agent.sample(context, previous_reward))       # already 0..K-1
        if not 0 <= action < agent.n_actions:
            raise ValueError(
                f"Agent returned action {action}, expected 0..{agent.n_actions - 1}"
            )
        return action

    def sample(self, reward) -> tuple[Array, Array]:
        """
        Samples the agent to the transmission matrix.

        Returns
        -------
        tuple
            The transmission matrices and tx_power indices (one per link).
        """

        self.step += 1
        self.rewards.append(reward)

        # loop invariant: after the append, reward of step t sits at index t-1, so
        # self.rewards[last_step[agent]] is the reward that followed the agent's last action.

        sharing_ap = np.random.choice(self.access_points).item()
        sharing_sta = np.random.choice(self.associations[sharing_ap]).item()

        # ---------------- level 1: AP group ---------------- #
        context_lvl1 = self.encoded_sharing_ap(sharing_ap, sharing_sta)

        # 1 -> AP participates, 0 -> AP does not (sharing AP always participates)
        find_groups_agent_action = np.zeros(shape=self.access_points.shape, dtype=np.int32)
        find_groups_agent_action[sharing_ap] = 1

        aps_taking_action = self.access_points[self.access_points != sharing_ap]

        for ap in aps_taking_action:
            ap = ap.item()
            action = self._sample(
                self.find_groups_agent[ap],
                context_lvl1,
                self.find_groups_agent_last_step[ap],
            )
            find_groups_agent_action[ap] = action
            self.find_groups_agent_last_action[ap] = action
            self.find_groups_agent_last_step[ap] = self.step

        # ---------------- level 2: station assignment ---------------- #
        context_lvl2 = find_groups_agent_action
        selected_aps = self.access_points[context_lvl2.astype(bool)]  # np.bool removed in NumPy>=1.24

        selected_ap_group = selected_aps[selected_aps != sharing_ap]

        # relative index of the station in associations[ap] (0-based)
        ap_sta_pairs = {
            int(ap): self._sample(
                self.assign_stations_agent[ap],
                context_lvl2,
                self.assign_stations_agent_last_step[ap],
            )
            for ap in selected_ap_group
        }

        for ap, sta_idx in ap_sta_pairs.items():
            self.assign_stations_agent_last_step[ap] = self.step
            self.assign_stations_agent_last_action[ap] = sta_idx

        ap_sta_pairs[sharing_ap] = (self.associations[sharing_ap] == sharing_sta).argmax().item()

        # ---------------- level 3: link combination ---------------- #
        context_lvl3 = self.encode_ap_stations_to_tx_vector(ap_sta_pairs)

        # 0-based index into link_comb_index_to_links
        ap_sta_links = {
            int(ap): self._sample(
                self.assign_links_agent[ap],
                context_lvl3,
                self.assign_links_agent_last_step[ap],
            )
            for ap in selected_aps
        }

        for ap, link_idx in ap_sta_links.items():
            self.assign_links_agent_last_step[ap] = self.step
            self.assign_links_agent_last_action[ap] = link_idx

        # ap -> sta
        sta_link_indices = {}
        for ap, link_idx in ap_sta_links.items():
            sta_selected_rel_index = ap_sta_pairs[ap]
            sta_index = self.associations[ap][sta_selected_rel_index]
            sta_link_indices[sta_index] = link_idx

        # ---------------- level 4: tx power ---------------- #
        context_lvl4 = self.encode_sta_links_vector(sta_link_indices)

        sta_links = {
            sta: self.link_comb_index_to_links[link_index]
            for sta, link_index in sta_link_indices.items()
        }

        link_ap_sta = {
            link: {
                "tx_matrix": np.zeros((self.n_nodes, self.n_nodes)),
                "tx_power_indices": np.zeros(self.n_nodes, dtype=np.int32)
            }
            for link in range(self.n_links)
        }

        for sta, links in sta_links.items():
            for link in links:
                tx_power_index = self._sample(
                    self.assign_tx_power_agent[sta, link],
                    context_lvl4,
                    self.assign_tx_power_agent_last_step[sta, link],
                )
                link_ap_sta[link]["tx_matrix"][self.inv_associations[sta], sta] = 1
                link_ap_sta[link]["tx_power_indices"][self.inv_associations[sta]] = tx_power_index
                self.assign_tx_power_agent_last_action[sta, link] = tx_power_index
                self.assign_tx_power_agent_last_step[sta, link] = self.step

        tx_matrices = np.array([link_ap_sta[r]["tx_matrix"] for r in range(self.n_links)], dtype=np.int16)
        tx_power_indices = np.array([link_ap_sta[r]["tx_power_indices"] for r in range(self.n_links)], dtype=np.int16)

        if self.logger is not None:
            self.logger.log(
                step=self.step,
                tx_matrices=tx_matrices,
                tx_power_indices=tx_power_indices,
                reward=self.rewards[-1],  # reward is 1 step delayed
            )

        return tx_matrices, tx_power_indices