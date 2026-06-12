"""Board basics: placement, groups, liberties, hashing."""

import pytest

from goengine import Board, Color


def test_board_sizes_supported() -> None:
    for size in (9, 13, 19):
        board = Board(size)
        assert board.size == size
        assert sum(1 for _ in board.points()) == size * size


def test_invalid_board_size_rejected() -> None:
    with pytest.raises(ValueError):
        Board(1)
    with pytest.raises(ValueError):
        Board(26)


def test_place_and_get() -> None:
    board = Board(9)
    board.place_stone((4, 4), Color.BLACK)
    assert board.get((4, 4)) is Color.BLACK
    assert board.get((0, 0)) is Color.EMPTY


def test_cannot_place_on_occupied_point() -> None:
    board = Board(9)
    board.place_stone((4, 4), Color.BLACK)
    with pytest.raises(ValueError):
        board.place_stone((4, 4), Color.WHITE)


def test_group_and_liberties() -> None:
    board = Board(9)
    board.place_stone((2, 2), Color.BLACK)
    board.place_stone((2, 3), Color.BLACK)
    stones, liberties = board.group_at((2, 2))
    assert stones == {(2, 2), (2, 3)}
    assert liberties == {(1, 2), (1, 3), (3, 2), (3, 3), (2, 1), (2, 4)}


def test_corner_liberties() -> None:
    board = Board(9)
    board.place_stone((0, 0), Color.WHITE)
    _, liberties = board.group_at((0, 0))
    assert liberties == {(0, 1), (1, 0)}


def test_position_hash_is_placement_only_and_reversible() -> None:
    board = Board(9)
    empty_hash = board.position_hash
    board.place_stone((4, 4), Color.BLACK)
    h1 = board.position_hash
    assert h1 != empty_hash
    board._set((4, 4), Color.EMPTY)
    assert board.position_hash == empty_hash
    # Same placement order-independence
    board.place_stone((1, 1), Color.BLACK)
    board.place_stone((2, 2), Color.WHITE)
    h_ab = board.position_hash
    board2 = Board(9)
    board2.place_stone((2, 2), Color.WHITE)
    board2.place_stone((1, 1), Color.BLACK)
    assert board2.position_hash == h_ab
