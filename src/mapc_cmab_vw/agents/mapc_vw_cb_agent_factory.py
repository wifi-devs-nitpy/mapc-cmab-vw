from itertools import chain, combinations
from typing import Iterable

import numpy as np

from mapc_cmab_vw.agents.mapc_agent import MapcAgent
from mapc_cmab_vw.agents.vw_contextual_bandit import VwCBAgent
from mapc_cmab_vw.agents.hierarchical_mapc_vw_cb import HierarchicalMapcVwCB


class MapcVwCBAgentFactory:
    """
    Factory for the hierarchical MAPC Vowpal Wabbit contextual-bandit agent.

    The factory mirrors the hierarchical DQN factory's four-level structure
    and context encoders. All public actions remain 0-based.
    """

    def __init__(
        self,
        associations: dict[int, list[int]],
        agent_params_lvl1: dict | None = None,
        agent_params_lvl2: dict | None = None,
        agent_params_lvl3: dict | None = None,
        agent_params_lvl4: dict | None = None,
        n_links: int = 3,
        n_tx_power_levels: int = 4,
        seed: int = 42,
        logger=None,
    ):
        self.associations = {
            ap: np.asarray(stations) for ap, stations in associations.items()
        }
        self.agent_params_lvl1 = agent_params_lvl1 or {}
        self.agent_params_lvl2 = agent_params_lvl2 or {}
        self.agent_params_lvl3 = agent_params_lvl3 or {}
        self.agent_params_lvl4 = agent_params_lvl4 or {}

        self.n_links = n_links
        self.n_tx_power_levels = n_tx_power_levels
        self.seed = seed
        self.logger = logger

        self.access_points = list(self.associations.keys())
        self.stations = list(
            chain.from_iterable(self.associations.values())
        )
        self.n_ap = len(self.access_points)
        self.n_sta = len(self.stations)
        self.stations_per_ap = len(self.associations[self.access_points[0]])

        self.ap_to_idx = {
            ap: i for i, ap in enumerate(self.access_points)
        }

        self.link_comb_index_to_links = {
            idx: list(link_comb)
            for idx, link_comb in enumerate(
                self._powerset_without_emptyset(range(self.n_links))
            )
        }

        self.sta_index_mapping = {
            sta: i for i, sta in enumerate(self.stations)
        }

    def create_hierarchical_vw_cb_agent(
        self,
        logger=None,
    ) -> MapcAgent:
        rng = np.random.default_rng(self.seed)

        # ---------------------------------------------------------------
        # LEVEL 1: AP participation
        # Context = sharing AP + sharing station encoding
        # Action = 0/1
        # ---------------------------------------------------------------
        lvl1_context_size = self.n_ap + self.stations_per_ap
        lvl1 = {
            ap: VwCBAgent(
                n_actions=2,
                context_size=lvl1_context_size,
                seed=int(rng.integers(0, 2**31 - 1)),
                **self.agent_params_lvl1,
            )
            for ap in self.access_points
        }

        # ---------------------------------------------------------------
        # LEVEL 2: station selection
        # Context = AP-group binary vector
        # Action = relative station index
        # ---------------------------------------------------------------
        lvl2 = {
            ap: VwCBAgent(
                n_actions=len(self.associations[ap]),
                context_size=self.n_ap,
                seed=int(rng.integers(0, 2**31 - 1)),
                **self.agent_params_lvl2,
            )
            for ap in self.access_points
        }

        # ---------------------------------------------------------------
        # LEVEL 3: link-combination selection
        # Context = flattened AP x station assignment matrix
        # Action = non-empty link-combination index
        # ---------------------------------------------------------------
        n_link_actions = 2**self.n_links - 1
        lvl3 = {
            ap: VwCBAgent(
                n_actions=n_link_actions,
                context_size=self.n_ap * self.stations_per_ap,
                seed=int(rng.integers(0, 2**31 - 1)),
                **self.agent_params_lvl3,
            )
            for ap in self.access_points
        }

        # ---------------------------------------------------------------
        # LEVEL 4: TX-power selection
        # Context = flattened STA x link activity matrix
        # Action = TX-power index
        # ---------------------------------------------------------------
        lvl4_context_size = self.n_sta * self.n_links

        lvl4 = {
            (int(sta), int(link)): VwCBAgent(
                n_actions=self.n_tx_power_levels,
                context_size=lvl4_context_size,
                seed=int(rng.integers(0, 2**31 - 1)),
                **self.agent_params_lvl4,
            )
            for sta in self.stations
            for link in range(self.n_links)
        }

        return HierarchicalMapcVwCB(
            associations=self.associations,
            find_groups_agent=lvl1,
            assign_stations_agent=lvl2,
            assign_links_agent=lvl3,
            assign_tx_power_agent=lvl4,
            encode_sharing_ap=self._encode_sharing_ap,
            encode_ap_group=self._encode_ap_group,
            encode_ap_stations_to_tx_vector=self._encode_ap_stations_to_tx_vector,
            encode_sta_links_vector=self._encode_sta_links_vector,
            link_comb_index_to_links=self.link_comb_index_to_links,
            n_links=self.n_links,
            n_tx_power_levels=self.n_tx_power_levels,
            logger=logger if logger is not None else self.logger,
        )

    def _encode_sharing_ap(self, sharing_ap, sharing_station):
        arr1 = np.isin(
            np.asarray(self.access_points),
            np.asarray(sharing_ap),
        ).astype(np.int32)

        arr2 = np.zeros(self.stations_per_ap, dtype=np.int32)
        rel_idx = int(
            np.argmax(self.associations[sharing_ap] == sharing_station)
        )
        arr2[rel_idx] = 1

        return np.concatenate([arr1, arr2])

    def _encode_ap_group(self, sharing_ap, selected_ap_group):
        ap_group = np.asarray(
            list(selected_ap_group) + [sharing_ap]
        )
        return np.isin(
            np.asarray(self.access_points),
            ap_group,
        ).astype(np.int32)

    def _encode_ap_stations_to_tx_vector(self, ap_sta_dict):
        result = np.zeros(
            (self.n_ap, self.stations_per_ap),
            dtype=np.int32,
        )

        rows = [
            self.ap_to_idx[ap]
            for ap in ap_sta_dict.keys()
        ]
        cols = list(ap_sta_dict.values())

        result[rows, cols] = 1
        return result.reshape(-1)

    def _encode_sta_links_vector(self, sta_links):
        result = np.zeros(
            (self.n_sta, self.n_links),
            dtype=np.int32,
        )

        rows = [
            self.sta_index_mapping[sta]
            for sta, idx in sta_links.items()
            for _ in self.link_comb_index_to_links[idx]
        ]
        cols = [
            link
            for idx in sta_links.values()
            for link in self.link_comb_index_to_links[idx]
        ]

        result[rows, cols] = 1
        return result.reshape(-1)

    @staticmethod
    def _powerset(iterable: Iterable):
        s = sorted(iterable)
        return chain.from_iterable(
            combinations(s, r)
            for r in range(len(s) + 1)
        )

    @staticmethod
    def _powerset_without_emptyset(iterable: Iterable):
        s = sorted(iterable)
        return chain.from_iterable(
            combinations(s, r)
            for r in range(1, len(s) + 1)
        )
