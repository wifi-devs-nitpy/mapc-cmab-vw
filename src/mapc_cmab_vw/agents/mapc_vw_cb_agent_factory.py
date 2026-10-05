from itertools import chain, combinations
from typing import Iterable

import numpy as np
from chex import Array

from mapc_cmab_vw.agents.mapc_agent import MapcAgent
from mapc_cmab_vw.agents.hierarchical_mapc_vw_cb import HierarchicalMapcVWAgent
from mapc_cmab_vw.agents.vw_contextual_bandit import VwCBAgent


class MapcVWAgentFactory:
    """
    Factory creating the hierarchical Vowpal Wabbit contextual-bandit agent.
    Mirrors ``MapcDQNAgentFactory`` (same sizes, same contexts, same encoders).

    Parameters
    ----------
    associations : dict[int, list[int]]
        Associations between APs and stations.
    agent_params_lvl1..4 : dict
        Per-level keyword overrides for ``VwCBAgent`` (e.g. {"epsilon": 0.1}).
        They are applied last, exactly like ``**self.agent_params_lvlX`` in the DQN factory.
    n_links : int
        Number of radio links.
    n_tx_power_levels : int
        Number of transmission power levels.
    seed : int
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            agent_params_lvl1: dict,
            agent_params_lvl2: dict,
            agent_params_lvl3: dict,
            agent_params_lvl4: dict,
            n_links: int = 3,
            n_tx_power_levels: int = 4,
            seed: int = 42,
            logger=None
    ):
        self.associations = {ap: np.asarray(stations) for ap, stations in associations.items()}
        self.agent_params_lvl1 = agent_params_lvl1
        self.agent_params_lvl2 = agent_params_lvl2
        self.agent_params_lvl3 = agent_params_lvl3
        self.agent_params_lvl4 = agent_params_lvl4
        self.n_tx_power_levels = n_tx_power_levels
        self.n_links = n_links
        self.seed = seed
        self.logger = logger

        np.random.seed(self.seed)

        self.inv_associations = {sta: ap for ap in associations.keys() for sta in associations[ap]}
        self.access_points = list(associations.keys())
        self.stations = list(chain.from_iterable(associations.values()))
        self.n_ap = len(self.access_points)
        self.n_sta = len(self.stations)
        self.n_nodes = self.n_ap + self.n_sta
        self.stations_per_ap = len(self.associations[0])  # assuming equal number of stations per AP
        self.ap_to_idx = {ap: i for i, ap in enumerate(self.access_points)}
        self.link_comb_index_to_links = {
            idx: list(link_comb)
            for idx, link_comb in enumerate(self._powerset_without_emptyset(range(self.n_links)))
        }
        self.sta_index_mapping = {sta: index for index, sta in enumerate(self.stations)}

    def create_hierarchical_VW_cmapc_agent(self, logger=None) -> MapcAgent:
        """
        Initialises the hierarchical VW agent.

        Number of actions per level (``n_actions``) is what VW needs; VW actions are
        1..n_actions and are shifted to 0..n_actions-1 inside HierarchicalMapcVWAgent._sample.
        """
        self.seed += 1
        np.random.seed(self.seed)

        # distinct deterministic seed per VW agent (otherwise all share one RNG stream)
        seed_counter = iter(range(self.seed * 100_000, self.seed * 100_000 + 10_000_000))

        def make(n_actions: int, context_size: int, overrides: dict) -> VwCBAgent:
            params = {"epsilon": 0.15, "seed": next(seed_counter), **overrides}
            return VwCBAgent(n_actions=n_actions, context_size=context_size, **params)

        # ---------- level 1: participate (action 1) / not participate (action 0), 0-based ----------
        # context: one-hot sharing AP + one-hot sharing station (relative idx)
        action_size_lvl1 = 2
        find_groups_agent = {
            ap: make(action_size_lvl1, self.n_ap + self.stations_per_ap, self.agent_params_lvl1)
            for ap in self.access_points
        }

        # ---------- level 2: station assignment ----------
        # context: encoded AP group (n_ap,)
        assign_stations_agent = {
            ap: make(len(self.associations[ap]), self.n_ap, self.agent_params_lvl2)
            for ap in self.access_points
        }

        # ---------- level 3: link combination ----------
        # context: tx vector over (ap, relative station) -> n_ap * stations_per_ap
        action_size_lvl3 = 2 ** self.n_links - 1
        assign_links_agent = {
            ap: make(action_size_lvl3, self.n_ap * self.stations_per_ap, self.agent_params_lvl3)
            for ap in self.access_points
        }

        # ---------- level 4: tx power per (station, link) ----------
        # context: station x link usage matrix, flattened
        grids = np.meshgrid(self.stations, list(range(self.n_links)), indexing="ij")
        sta_link = np.stack([grid.ravel() for grid in grids], axis=-1)

        assign_tx_power_agent = {
            (int(sta), int(link)): make(
                self.n_tx_power_levels, self.n_sta * self.n_links, self.agent_params_lvl4
            )
            for sta, link in sta_link
        }

        return HierarchicalMapcVWAgent(
            associations=self.associations,
            find_groups_agent=find_groups_agent,
            assign_stations_agent=assign_stations_agent,
            assign_links_agent=assign_links_agent,
            assign_tx_power_agent=assign_tx_power_agent,
            encode_sharing_ap=self._encode_sharing_ap,
            encode_ap_group=self._encode_ap_group,
            encode_ap_stations_to_tx_vector=self._encode_ap_stations_to_tx_vector,
            encode_sta_links_vector=self._encode_sta_links_vector,
            ap_group_action_to_ap_group=self._ap_group_action_to_ap_group,
            link_comb_index_to_links=self.link_comb_index_to_links,
            sta_index_mapping=self.sta_index_mapping,
            n_links=self.n_links,
            n_tx_power_levels=self.n_tx_power_levels,
            logger=logger if logger is not None else self.logger
        )

    # ------------------------------------------------------------------ #
    # Helpers / encoders: unchanged from the DQN factory
    # ------------------------------------------------------------------ #
    @staticmethod
    def _powerset(iterable: Iterable) -> Iterable:
        s = sorted(list(iterable))
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    @staticmethod
    def _powerset_without_emptyset(iterable: Iterable) -> Iterable:
        s = sorted(list(iterable))
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    def _ap_group_action_to_ap_group(self, ap_group_action: int, sharing_ap: int) -> tuple[int]:
        ap_set = set(self.access_points).difference({sharing_ap})
        return tuple(self._powerset(ap_set))[ap_group_action]

    def _sta_group_action_to_sta_group(self, sta_group_action: dict[int, int]) -> list[int]:
        return [self.associations[ap][sta_id] for ap, sta_id in sta_group_action.items()]

    def _encode_ap_group(self, sharing_ap, selected_ap_group) -> Array:
        ap_group = np.asarray(list(selected_ap_group) + [sharing_ap])
        return np.isin(np.asarray(self.access_points), ap_group).astype(np.int32)

    def _encode_ap_stations_to_tx_vector(self, ap_sta_dict):
        res = np.zeros(shape=(self.n_ap, self.stations_per_ap))
        ap_to_idx = self.ap_to_idx
        rows = [ap_to_idx[ap] for ap in ap_sta_dict.keys()]
        cols = list(ap_sta_dict.values())
        res[rows, cols] = 1
        return res.reshape(-1)

    def _encode_sta_links_vector(self, sta_links: dict[int, int]):
        res = np.zeros((self.n_sta, self.n_links))

        sta_index_mapping = self.sta_index_mapping
        rows = [sta_index_mapping[sta] for sta, idx in sta_links.items()
                for _ in self.link_comb_index_to_links[idx]]
        cols = [link for idx in sta_links.values()
                for link in self.link_comb_index_to_links[idx]]

        res[rows, cols] = 1
        return res.reshape(-1)

    def _encode_sharing_ap(self, sharing_ap, sharing_station) -> Array:
        arr1 = np.isin(np.asarray(self.access_points), np.asarray(sharing_ap)).astype(np.int32)
        arr2 = np.zeros(self.stations_per_ap)
        rel_index_of_sharing_station = (self.associations[sharing_ap] == sharing_station).argmax()
        arr2[rel_index_of_sharing_station] = 1

        return np.concatenate([arr1, arr2], axis=0)
