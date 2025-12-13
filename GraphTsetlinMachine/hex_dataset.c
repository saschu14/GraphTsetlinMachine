// hex_dataset.c
// Generates Hex game states (full, -2 moves, -5 moves) and writes them as CSV.
// Assumes BOARD_DIM is defined (e.g. 11).

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef BOARD_DIM
    #define BOARD_DIM 11
#endif

int neighbors[] = {
    -(BOARD_DIM + 2) + 1,
    -(BOARD_DIM + 2),
    -1,
    1,
    (BOARD_DIM + 2),
    (BOARD_DIM + 2) - 1
};

struct hex_game {
    int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
    int open_positions[BOARD_DIM*BOARD_DIM];
    int number_of_open_positions;
    int moves[BOARD_DIM*BOARD_DIM];
    int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void hg_init(struct hex_game *hg)
{
    for (int i = 0; i < BOARD_DIM+2; ++i) {
        for (int j = 0; j < BOARD_DIM+2; ++j) {
            int idx2 = (i*(BOARD_DIM + 2) + j) * 2;
            hg->board[idx2] = 0;
            hg->board[idx2 + 1] = 0;

            if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {
                hg->open_positions[(i-1)*BOARD_DIM + (j-1)] = i*(BOARD_DIM + 2) + j;
            }

            if (i == 0) {
                hg->connected[idx2] = 1;
            } else {
                hg->connected[idx2] = 0;
            }

            if (j == 0) {
                hg->connected[idx2 + 1] = 1;
            } else {
                hg->connected[idx2 + 1] = 0;
            }
        }
    }

    hg->number_of_open_positions = BOARD_DIM * BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position)
{
    hg->connected[position*2 + player] = 1;

    if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {
        return 1;
    }

    if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {
        return 1;
    }

    for (int i = 0; i < 6; ++i) {
        int neighbor = position + neighbors[i];
        if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
            if (hg_connect(hg, player, neighbor)) {
                return 1;
            }
        }
    }
    return 0;
}

int hg_winner(struct hex_game *hg, int player, int position)
{
    for (int i = 0; i < 6; ++i) {
        int neighbor = position + neighbors[i];
        if (hg->connected[neighbor*2 + player]) {
            return hg_connect(hg, player, position);
        }
    }
    return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player)
{
    int random_empty_position_index = rand() % hg->number_of_open_positions;

    int empty_position = hg->open_positions[random_empty_position_index];

    hg->board[empty_position * 2 + player] = 1;

    int move_index = BOARD_DIM*BOARD_DIM - hg->number_of_open_positions;
    hg->moves[move_index] = empty_position;

    hg->open_positions[random_empty_position_index] =
        hg->open_positions[hg->number_of_open_positions - 1];

    hg->number_of_open_positions--;

    return empty_position;
}

int hg_full_board(struct hex_game *hg)
{
    return hg->number_of_open_positions == 0;
}

void hg_clear_board(struct hex_game *hg)
{
    for (int i = 0; i < (BOARD_DIM+2)*(BOARD_DIM+2); ++i) {
        hg->board[i*2] = 0;
        hg->board[i*2 + 1] = 0;
    }
}

void hg_replay_prefix(struct hex_game *hg, int K)
{
    hg_clear_board(hg);

    for (int t = 0; t < K; ++t) {
        int player = t % 2;
        int pos = hg->moves[t];
        hg->board[pos * 2 + player] = 1;
    }
}

void hg_write_state_csv(FILE *f, struct hex_game *hg, int winner, int moves_left)
{
    fprintf(f, "%d,%d", winner, moves_left);

    for (int i = 1; i <= BOARD_DIM; ++i) {
        for (int j = 1; j <= BOARD_DIM; ++j) {
            int pos = i*(BOARD_DIM+2) + j;
            int p0 = hg->board[pos*2];
            int p1 = hg->board[pos*2 + 1];
            int val = 0;
            if (p0) val = 1;
            else if (p1) val = 2;
            fprintf(f, ",%d", val);
        }
    }
    fprintf(f, "\n");
}

int main(int argc, char **argv)
{
    int num_games = 10000;
    const char *filename = "hex_11x11_dataset.csv";

    if (argc >= 2) {
        num_games = atoi(argv[1]);
    }
    if (argc >= 3) {
        filename = argv[2];
    }

    printf("Generating %d games, writing to %s\n", num_games, filename);

    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("fopen");
        return 1;
    }

    fprintf(f, "winner,moves_left");
    for (int i = 0; i < BOARD_DIM*BOARD_DIM; ++i) {
        fprintf(f, ",c%d", i);
    }
    fprintf(f, "\n");

    srand((unsigned int)time(NULL));

    struct hex_game hg;

    for (int game = 0; game < num_games; ++game) {
        hg_init(&hg);

        int winner = -1;
        int player = 0;

        while (!hg_full_board(&hg)) {
            int position = hg_place_piece_randomly(&hg, player);

            if (hg_winner(&hg, player, position)) {
                winner = player;
                break;
            }

            player = 1 - player;
        }

        if (winner == -1) {
            continue;
        }

        int total_moves = BOARD_DIM*BOARD_DIM - hg.number_of_open_positions;

        int moves_left_values[] = {0, 2, 5};
        int num_levels = sizeof(moves_left_values) / sizeof(moves_left_values[0]);

        for (int idx = 0; idx < num_levels; ++idx) {
            int moves_left = moves_left_values[idx];
            int K = total_moves - moves_left;
            if (K <= 0) {
                continue;
            }

            hg_replay_prefix(&hg, K);
            hg_write_state_csv(f, &hg, winner, moves_left);
        }

        if ((game + 1) % 1000 == 0) {
            printf("Simulated %d games...\n", game + 1);
        }
    }

    fclose(f);
    printf("Done.\n");

    return 0;
}
