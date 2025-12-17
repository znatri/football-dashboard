class PlayerBallAssigner:
    def __init__(self):
        # Threshold in meters between player foot and ball for possession.
        self.max_player_ball_distance_m = 1.2

    def assigner_ball_to_player(self, players, ball_position_field):
        """
        Assign ball to nearest player in FIELD coordinates (meters).

        :param players: dict of player_id -> track_info, each containing 'position_field'
        :param ball_position_field: (X, Y) ball position in field coordinates (meters)
        :return: player_id that controls the ball or -1 if none within threshold.
        """
        from utils import measure_distance

        if ball_position_field is None:
            return -1

        min_distance = 99999.0
        assigned_player = -1

        for player_id, player in players.items():
            player_pos = player.get("position_field")
            if player_pos is None:
                continue

            distance = measure_distance(player_pos, ball_position_field)

            if (
                distance < self.max_player_ball_distance_m
                and distance < min_distance
            ):
                min_distance = distance
                assigned_player = player_id

        return assigned_player

