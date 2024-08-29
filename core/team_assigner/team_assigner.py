from typing import List, Dict, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """
    A class used to assign team colors to players based on their uniforms using KMeans clustering.
    """

    def __init__(self) -> None:
        """
        Initializes the TeamAssigner with empty dictionaries for team colors and player-team assignments,
        and sets the KMeans model to None.
        """
        self.team_colors: Dict[int, np.ndarray] = {}
        self.player_team_dict: Dict[int, int] = {}
        self.kmeans: KMeans = None

    def _get_clustering_model(self, image: np.ndarray) -> KMeans:
        """
        Creates and fits a KMeans clustering model to the given image.

        :param image: np.ndarray
            The image from which to extract the clustering model.

        :return: KMeans
            The fitted KMeans clustering model.
        """
        image_2d = image.reshape(-1, 3)
        num_clusters = 2
        kmeans = KMeans(n_clusters=num_clusters, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def _extract_player_color(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extracts the primary color of a player's uniform from a given frame and bounding box.

        :param frame: np.ndarray
            The image frame containing the player.
        :param bbox: Tuple[int, int, int, int]
            The bounding box coordinates (x1, y1, x2, y2) of the player.

        :return: np.ndarray
            The primary color of the player's uniform.
        """
        img = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        top_half_image = img[: img.shape[0] // 2, :]

        kmeans = self._get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]

        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_team_colors(
        self,
        frame: np.ndarray,
        player_detections: Dict[int, Dict[str, Union[int, List[int]]]],
    ) -> None:
        """
        Assigns team colors based on the detected player bounding boxes in the frame.

        :param frame: np.ndarray
            The image frame containing the players.
        :param player_detections: Dict[int, Dict[str, Union[int, List[int]]]]
            A dictionary containing player detections with their bounding boxes.
        """
        player_colors = [
            self._extract_player_color(frame, player_detection["bbox"])
            for player_detection in player_detections.values()
        ]

        num_clusters = 2
        kmeans = KMeans(n_clusters=num_clusters, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors = {
            1: kmeans.cluster_centers_[0],
            2: kmeans.cluster_centers_[1],
        }

    def get_player_team(
        self, frame: np.ndarray, player_bbox: Tuple[int, int, int, int], player_id: int
    ) -> int:
        """
        Determines the team of a player based on their uniform color.

        :param frame: np.ndarray
            The image frame containing the player.
        :param player_bbox: Tuple[int, int, int, int]
            The bounding box coordinates (x1, y1, x2, y2) of the player.
        :param player_id: int
            The unique identifier of the player.
        :return: int
            The team ID to which the player belongs.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self._extract_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0])
        team_id += 1

        # todo: fix this - Hard assign goalkeeper for now
        if player_id == 220:
            team_id = 1
        if player_id == 630:
            team_id = 2

        self.player_team_dict[player_id] = team_id
        return team_id
