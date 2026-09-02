#ifndef ARC_SOLVE_H
#define ARC_SOLVE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arc/game.h"

/* Breadth-first search over a DSL game's states from its current level
 * start, using the arrows, ACTION5 when the game declares it, and clicks on
 * the cells whose starting kind responds to a click. Returns 1 when a win
 * or level advance was found (shortest_out = its depth), 0 when the whole
 * reachable graph was exhausted without one, -1 when max_nodes ran out. */
int32_t arc_dsl_solve(struct arc_game *game, int32_t max_nodes,
		      int32_t *shortest_out, int32_t *nodes_out);

#ifdef __cplusplus
}
#endif

#endif
