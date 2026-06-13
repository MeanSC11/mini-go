"""Manual smoke check: parallel self-play workers (process pool)."""

from azero.config import Config
from azero.selfplay import generate_games

if __name__ == "__main__":
    config = Config(
        board_size=5, history_planes=2, blocks=1, filters=8,
        simulations=4, temperature_moves=2, max_game_moves=20,
        workers=2, selfplay_device="cpu", selfplay_concurrent_games=2, device="cpu",
    )
    samples, stats = generate_games(None, config, total_games=2)
    print(f"OK: {len(samples)} samples from 2 games across 2 workers; "
          f"avg batch {stats.avg_batch:.1f}")
