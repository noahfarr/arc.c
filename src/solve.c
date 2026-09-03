#include <stdlib.h>
#include <string.h>

#include "arc/dsl.h"
#include "arc/solve.h"

struct action {
	int32_t id;
	int32_t x;
	int32_t y;
};

/* Open-addressed set of state hashes. */
struct set {
	uint64_t *keys;
	uint8_t *used;
	size_t mask;
};

static int set_insert(struct set *s, uint64_t key)
{
	size_t i = (size_t)(key ^ (key >> 29)) & s->mask;

	for (;;) {
		if (!s->used[i]) {
			s->used[i] = 1;
			s->keys[i] = key;
			return 1;
		}
		if (s->keys[i] == key)
			return 0;
		i = (i + 1) & s->mask;
	}
}

static int32_t actions_for(const struct arc_game *game, struct action *out,
			   int32_t cap)
{
	const struct arc_dsl_spec *s = (const struct arc_dsl_spec *)game->statics;
	int32_t level = game->engine.level_index;
	int32_t n = s->grid_w * s->grid_h;
	const int8_t *layout = s->layout + (size_t)level * n;
	uint8_t clickable[ARC_DSL_MAX_KINDS] = { 0 };
	int any_cell = 0;
	int32_t count = 0;

	for (int32_t a = 1; a <= 4; a++)
		out[count++] = (struct action){ a, 0, 0 };
	if (s->uses_action5)
		out[count++] = (struct action){ ARC_ACTION5, 0, 0 };
	for (int32_t k = 0; k < s->num_kinds; k++)
		if (s->kinds[k].on_click)
			clickable[k] = 1;
	for (int32_t r = 0; r < s->num_rules; r++) {
		const struct arc_dsl_rule *rule = &s->rules[r];

		if (!rule->enabled || rule->trigger != ARC_DSL_ON_CLICK)
			continue;
		if (rule->subject < 0)
			any_cell = 1;
		else
			clickable[rule->subject] = 1;
	}
	for (int32_t y = 0; y < s->grid_h; y++)
		for (int32_t x = 0; x < s->grid_w; x++) {
			int8_t k = layout[y * s->grid_w + x];

			if (count >= cap)
				return count;
			if (any_cell || (k >= 0 && clickable[k]))
				out[count++] = (struct action){
					ARC_ACTION6,
					s->origin_x + x * s->pitch + s->pitch / 2,
					s->origin_y + y * s->pitch + s->pitch / 2
				};
		}
	return count;
}

int32_t arc_dsl_solve_path(struct arc_game *game, int32_t max_nodes,
			   int32_t *path_out, int32_t path_cap,
			   int32_t *shortest_out, int32_t *nodes_out)
{
	const size_t aux_size = sizeof(struct arc_dsl_aux);
	const size_t size = arc_game_state_size(game, aux_size);
	struct action actions[4 + 1 + ARC_DSL_MAX_GRID * ARC_DSL_MAX_GRID];
	int32_t num_actions = actions_for(game, actions,
					  (int32_t)(sizeof actions / sizeof actions[0]));
	int32_t start_level = game->engine.level_index;
	size_t cap = 1;
	struct set seen;
	unsigned char *states;
	int32_t *depth;
	int32_t *parent;
	int32_t *via;
	int32_t head = 0, tail = 0, result = 0;

	while (cap < (size_t)max_nodes * 4)
		cap <<= 1;
	seen.mask = cap - 1;
	seen.keys = calloc(cap, sizeof(uint64_t));
	seen.used = calloc(cap, 1);
	states = malloc((size_t)(max_nodes + 1) * size);
	depth = malloc((size_t)(max_nodes + 1) * sizeof(int32_t));
	parent = malloc((size_t)(max_nodes + 1) * sizeof(int32_t));
	via = malloc((size_t)(max_nodes + 1) * sizeof(int32_t));
	*shortest_out = -1;

	arc_game_save(game, aux_size, states);
	depth[0] = 0;
	set_insert(&seen, arc_dsl_state_hash(game));
	tail = 1;

	while (head < tail) {
		const unsigned char *state = states + (size_t)head * size;
		int32_t d = depth[head];

		head++;
		for (int32_t a = 0; a < num_actions; a++) {
			arc_game_load(game, aux_size, state);
			arc_game_perform_action_frames(game, actions[a].id,
						       actions[a].x, actions[a].y,
						       NULL, 0);
			if (game->engine.status == WIN ||
			    game->engine.level_index > start_level) {
				*shortest_out = d + 1;
				result = 1;
				if (path_out && path_cap >= d + 1) {
					/* Walk parents back from the node we
					 * expanded; the final action is a. */
					int32_t n = head - 1, k = d;

					path_out[3 * k] = actions[a].id;
					path_out[3 * k + 1] = actions[a].x;
					path_out[3 * k + 2] = actions[a].y;
					while (k > 0) {
						k--;
						path_out[3 * k] = actions[via[n]].id;
						path_out[3 * k + 1] = actions[via[n]].x;
						path_out[3 * k + 2] = actions[via[n]].y;
						n = parent[n];
					}
				}
				goto done;
			}
			if (game->engine.status == GAME_OVER)
				continue;
			if (!set_insert(&seen, arc_dsl_state_hash(game)))
				continue;
			if (tail > max_nodes) {
				result = -1;
				goto done;
			}
			arc_game_save(game, aux_size, states + (size_t)tail * size);
			depth[tail] = d + 1;
			parent[tail] = head - 1;
			via[tail] = a;
			tail++;
		}
	}
done:
	*nodes_out = head;
	arc_game_load(game, aux_size, states);
	free(seen.keys);
	free(seen.used);
	free(states);
	free(depth);
	free(parent);
	free(via);
	return result;
}

int32_t arc_dsl_solve(struct arc_game *game, int32_t max_nodes,
		      int32_t *shortest_out, int32_t *nodes_out)
{
	return arc_dsl_solve_path(game, max_nodes, NULL, 0, shortest_out,
				  nodes_out);
}
