"""Fast CPU tests for the training stack (tiny board, tiny net)."""

import dataclasses

import numpy as np
import pytest
import torch

from goengine import Color, Game, Move

from azero.checkpoint import (
    build_network,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)
from azero.config import Config
from azero.features import HistoryTracker, encode, index_to_move, move_to_index
from azero.mcts import MCTS
from azero.replay import ReplayBuffer
from azero.selfplay import play_game


@pytest.fixture()
def tiny_config() -> Config:
    return Config(
        board_size=5,
        komi=7.5,
        history_planes=2,
        blocks=1,
        filters=8,
        simulations=8,
        eval_simulations=4,
        temperature_moves=4,
        max_game_moves=30,
        device="cpu",
    )


def test_feature_encoding_shapes_and_planes(tiny_config: Config) -> None:
    game = Game(5)
    tracker = HistoryTracker(tiny_config.history_planes, game.board)
    game.play(Move.play(2, 2))  # black
    tracker.push(game.board)
    planes = encode(game.board, tracker, game.current_player, 2)
    assert planes.shape == (5, 5, 5)
    # White to move: black stone at (2,2) is the opponent -> plane 2 (hist+0).
    assert planes[2, 2, 2] == 1.0
    assert planes[0, 2, 2] == 0.0
    # Turn plane: zeros because white is to move.
    assert planes[4].sum() == 0.0


def test_move_index_roundtrip() -> None:
    size = 9
    for move in [Move.play(0, 0), Move.play(8, 8), Move.play(3, 7), Move.pass_turn()]:
        index = move_to_index(move, size)
        back = index_to_move(index, size)
        assert back.is_pass == move.is_pass
        assert back.point == move.point


def test_network_forward_shapes(tiny_config: Config) -> None:
    net = build_network(tiny_config, "cpu")
    x = torch.zeros(3, tiny_config.input_planes, 5, 5)
    logits, value = net(x)
    assert logits.shape == (3, 26)  # 25 points + pass
    assert value.shape == (3,)
    assert torch.all(value.abs() <= 1.0)


def test_mcts_produces_legal_visit_distribution(tiny_config: Config) -> None:
    net = build_network(tiny_config, "cpu")
    net.eval()
    game = Game(5, komi=7.5)
    tracker = HistoryTracker(tiny_config.history_planes, game.board)
    mcts = MCTS(net, tiny_config)
    policy, value = mcts.run(game, tracker, add_noise=True)
    assert policy.shape == (26,)
    assert abs(policy.sum() - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0
    move, _, _ = mcts.choose_move(game, tracker, temperature=1.0)
    assert game.is_legal(move)


def test_selfplay_generates_consistent_samples(tiny_config: Config) -> None:
    net = build_network(tiny_config, "cpu")
    net.eval()
    samples = play_game(net, tiny_config)
    assert samples
    states, policy, z = samples[0]
    assert states.shape == (tiny_config.input_planes, 5, 5)
    assert policy.shape == (26,)
    assert z in (-1.0, 0.0, 1.0)
    # Outcomes alternate perspective: consecutive samples have opposite z
    # (no draws on a 5x5 board with komi 7.5).
    zs = [s[2] for s in samples]
    assert all(abs(a + b) < 1e-9 for a, b in zip(zs, zs[1:]))


def test_mcts_passes_when_only_own_eyes_remain(tiny_config: Config) -> None:
    # Board fully controlled by black except its own eyes -> MCTS must pass,
    # not fill its own territory (candidate moves exclude eye fills).
    net = build_network(tiny_config, "cpu")
    net.eval()
    game = Game(5, komi=7.5)
    for r in range(5):
        for c in range(5):
            if (r, c) not in {(1, 1), (3, 3)}:
                game.board._set((r, c), Color.BLACK)
    game._position_history = {game.board.position_hash}
    game.current_player = Color.BLACK
    tracker = HistoryTracker(tiny_config.history_planes, game.board)
    mcts = MCTS(net, tiny_config)
    move, _, _ = mcts.choose_move(game, tracker, temperature=0.0, add_noise=False)
    assert move.is_pass


def test_replay_buffer_roundtrip(tmp_path, tiny_config: Config) -> None:
    buffer = ReplayBuffer(capacity=100)
    rng = np.random.default_rng(0)
    samples = [
        (
            rng.random((tiny_config.input_planes, 5, 5)).astype(np.float32),
            rng.random(26).astype(np.float32),
            1.0,
        )
        for _ in range(10)
    ]
    buffer.add(samples)
    states, policies, values = buffer.sample(4)
    assert states.shape[0] == 4
    path = tmp_path / "buffer.npz"
    buffer.save(path)
    restored = ReplayBuffer(capacity=100)
    restored.load(path)
    assert len(restored) == 10


def test_checkpoint_roundtrip(tmp_path, tiny_config: Config) -> None:
    net = build_network(tiny_config, "cpu")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(net, tiny_config, 7, path)
    restored, config, iteration = load_checkpoint(path, "cpu")
    assert iteration == 7
    assert config.board_size == 5
    for a, b in zip(net.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def test_alphazero_player_via_checkpoint(tmp_path, tiny_config: Config) -> None:
    from azero.bot import AlphaZeroPlayer

    net = build_network(tiny_config, "cpu")
    path = tmp_path / "ckpt.pt"
    save_checkpoint(net, tiny_config, 1, path)
    player = AlphaZeroPlayer(str(path), simulations=8, device="cpu")
    game = Game(5, komi=7.5)
    game.play(Move.play(2, 2))
    move, win_rate, policy = player.search(game)
    assert game.is_legal(move)
    assert win_rate is not None and 0.0 <= win_rate <= 1.0
    assert policy


def test_resolve_device_falls_back() -> None:
    assert resolve_device("cpu") == "cpu"
    if not torch.cuda.is_available():
        assert resolve_device("cuda") == "cpu"
