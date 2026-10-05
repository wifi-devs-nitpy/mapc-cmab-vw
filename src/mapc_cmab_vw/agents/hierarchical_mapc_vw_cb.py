from collections import defaultdict
from itertools import chain
from typing import Callable

import numpy as np
from chex import Array

from mapc_cmab_vw.agents.mapc_agent import MapcAgent


class HierarchicalMapcVwCB(MapcAgent):
    """
    Hierarchical MAPC agent using Vowpal Wabbit contextual-bandit agents.

    Hierarchy:
      Level 1: decide which APs participate.
      Level 2: each selected AP chooses a station (relative station index).
      Level 3: each selected AP chooses a non-empty link combination.
      Level 4: each selected (station, link) chooses a TX-power index.

    IMPORTANT:
    Every VwCBAgent exposes actions using the same 0-based convention as DQN.
    The 0-based -> 1-based conversion is internal to VwCBAgent only.
    Therefore this class contains no VW-specific +1/-1 action arithmetic.
    """

    def __init__(
        self,
        associations: dict[int, list[int]],
        find_groups_agent: dict[int, object],
        assign_stations_agent: dict[int, object],
        assign_links_agent: dict[int, object],
        assign_tx_power_agent: dict[tuple[int, int], object],
        encode_sharing_ap: Callable,
        encode_ap_group: Callable,
        encode_ap_stations_to_tx_vector: Callable,
        encode_sta_links_vector: Callable,
        link_comb_index_to_links: dict[int, list],
        n_links: int,
        n_tx_power_levels: int,
        logger=None,
    ):
        self.associations = {
            ap: np.asarray(stations) for ap, stations in associations.items()
        }
        self.inv_associations = {
            sta: ap for ap in self.associations for sta in self.associations[ap]
        }

        self.find_groups_agent = find_groups_agent
        self.assign_stations_agent = assign_stations_agent
        self.assign_links_agent = assign_links_agent
        self.assign_tx_power_agent = assign_tx_power_agent

        self.encoded_sharing_ap = encode_sharing_ap
        self.encode_ap_group = encode_ap_group
        self.encode_ap_stations_to_tx_vector = encode_ap_stations_to_tx_vector
        self.encode_sta_links_vector = encode_sta_links_vector

        self.link_comb_index_to_links = link_comb_index_to_links
        self.n_links = n_links
        self.n_tx_power_levels = n_tx_power_levels
        self.logger = logger

        self.access_points = np.asarray(list(self.associations.keys()))
        self.stations = np.asarray(
            list(chain.from_iterable(self.associations.values()))
        )
        self.n_ap = len(self.access_points)
        self.n_sta = len(self.stations)
        self.n_nodes = self.n_ap + self.n_sta
        self.stations_per_ap = len(self.associations[self.access_points[0]])

        # Per-agent delayed-transition bookkeeping.
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

    def sample(self, reward) -> tuple[Array, Array]:
        self.step += 1
        self.rewards.append(reward)

        # ---------------------------------------------------------------
        # LEVEL 1: choose a sharing AP + select the other participating APs
        # ---------------------------------------------------------------
        sharing_ap = int(np.random.choice(self.access_points))
        sharing_sta = int(np.random.choice(self.associations[sharing_ap]))

        context_lvl1 = self.encoded_sharing_ap(sharing_ap, sharing_sta)

        find_groups_agent_action = np.zeros(
            shape=self.access_points.shape,
            dtype=np.int32,
        )

        # The DQN architecture represents AP participation as:
        #   0 -> not participating
        #   1 -> participating
        # Keep this convention unchanged.
        find_groups_agent_action[self.access_points == sharing_ap] = 1

        aps_taking_action = self.access_points[self.access_points != sharing_ap]

        for ap in aps_taking_action:
            ap = int(ap)

            action = self.find_groups_agent[ap].sample(
                context_lvl1,
                self._previous_reward(
                    self.find_groups_agent_last_step[ap]
                ),
            )
            find_groups_agent_action[ap] = action

            self.find_groups_agent_last_action[ap] = action
            self.find_groups_agent_last_step[ap] = self.step

        context_lvl2 = find_groups_agent_action
        selected_aps = self.access_points[
            context_lvl2.astype(bool)
        ]
        selected_ap_group = selected_aps[selected_aps != sharing_ap]

        # ---------------------------------------------------------------
        # LEVEL 2: each selected AP chooses one of its associated STAs
        # ---------------------------------------------------------------
        ap_sta_pairs = {}

        for ap in selected_ap_group:
            ap = int(ap)
            action_size = len(self.associations[ap])

            action = self.assign_stations_agent[ap].sample(
                context_lvl2,
                self._previous_reward(
                    self.assign_stations_agent_last_step[ap]
                ),
            )

            if not 0 <= action < action_size:
                raise RuntimeError(
                    f"Invalid level-2 action {action} for AP {ap}; "
                    f"expected [0, {action_size - 1}]"
                )

            ap_sta_pairs[ap] = int(action)
            self.assign_stations_agent_last_step[ap] = self.step
            self.assign_stations_agent_last_action[ap] = int(action)

        # Sharing AP's station is fixed by the DCF/contention decision.
        ap_sta_pairs[sharing_ap] = int(
            np.argmax(self.associations[sharing_ap] == sharing_sta)
        )

        # ---------------------------------------------------------------
        # LEVEL 3: choose link-combination for each selected AP
        # ---------------------------------------------------------------
        context_lvl3 = self.encode_ap_stations_to_tx_vector(ap_sta_pairs)

        ap_sta_links = {}

        for ap in selected_aps:
            ap = int(ap)

            action = self.assign_links_agent[ap].sample(
                context_lvl3,
                self._previous_reward(
                    self.assign_links_agent_last_step[ap]
                ),
            )

            if not 0 <= action < len(self.link_comb_index_to_links):
                raise RuntimeError(
                    f"Invalid level-3 action {action} for AP {ap}"
                )

            ap_sta_links[ap] = int(action)
            self.assign_links_agent_last_step[ap] = self.step
            self.assign_links_agent_last_action[ap] = int(action)

        # Convert relative AP -> STA link choice into STA -> link choice.
        sta_link_indices = {}
        for ap, link_idx in ap_sta_links.items():
            sta_rel_idx = ap_sta_pairs[ap]
            sta = int(self.associations[ap][sta_rel_idx])
            sta_link_indices[sta] = link_idx

        # ---------------------------------------------------------------
        # LEVEL 4: choose TX power for each active STA-link
        # ---------------------------------------------------------------
        context_lvl4 = self.encode_sta_links_vector(sta_link_indices)

        link_ap_sta = {
            link: {
                "tx_matrix": np.zeros(
                    (self.n_nodes, self.n_nodes), dtype=np.int16
                ),
                "tx_power_indices": np.zeros(
                    self.n_nodes, dtype=np.int16
                ),
            }
            for link in range(self.n_links)
        }

        for sta, links in (
            (sta, self.link_comb_index_to_links[link_idx])
            for sta, link_idx in sta_link_indices.items()
        ):
            for link in links:
                key = (sta, link)

                action = self.assign_tx_power_agent[key].sample(
                    context_lvl4,
                    self._previous_reward(
                        self.assign_tx_power_agent_last_step[key]
                    ),
                )

                if not 0 <= action < self.n_tx_power_levels:
                    raise RuntimeError(
                        f"Invalid level-4 action {action} for "
                        f"(STA={sta}, link={link})"
                    )

                ap = self.inv_associations[sta]

                link_ap_sta[link]["tx_matrix"][ap, sta] = 1
                link_ap_sta[link]["tx_power_indices"][ap] = action

                self.assign_tx_power_agent_last_action[key] = int(action)
                self.assign_tx_power_agent_last_step[key] = self.step

        tx_matrices = np.asarray(
            [link_ap_sta[r]["tx_matrix"] for r in range(self.n_links)],
            dtype=np.int16,
        )
        tx_power_indices = np.asarray(
            [link_ap_sta[r]["tx_power_indices"] for r in range(self.n_links)],
            dtype=np.int16,
        )

        if self.logger is not None:
            self.logger.log(
                step=self.step,
                tx_matrices=tx_matrices,
                tx_power_indices=tx_power_indices,
                reward=self.rewards[-1],
            )

        return tx_matrices, tx_power_indices

    def _previous_reward(self, last_step: int) -> float:
        """
        Existing simulator convention:
        reward for action at `last_step` is stored at rewards[last_step].
        A zero last_step means there is no real previous transition yet.
        """
        if last_step == 0:
            return 0.0
        return float(self.rewards[last_step])
