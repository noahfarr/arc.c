#include <string.h>

#include "arc/dsl.h"

static const int32_t DX[4] = { 0, 0, -1, 1 };
static const int32_t DY[4] = { -1, 1, 0, 0 };

static const struct arc_dsl_spec *spec_of(const struct arc_game *game)
{
	return (const struct arc_dsl_spec *)game->statics;
}

static int32_t idx(const struct arc_dsl_spec *s, int32_t x, int32_t y)
{
	return y * s->grid_w + x;
}

static int inside(const struct arc_dsl_spec *s, int32_t x, int32_t y)
{
	return x >= 0 && y >= 0 && x < s->grid_w && y < s->grid_h;
}

void arc_dsl_zero_aux(void *aux)
{
	memset(aux, 0, sizeof(struct arc_dsl_aux));
}

int32_t arc_dsl_num_actions(const struct arc_dsl_spec *spec)
{
	return spec->grid_w * spec->grid_h;
}

static uint64_t fnv(uint64_t h, const void *data, size_t len)
{
	const unsigned char *p = (const unsigned char *)data;

	for (size_t i = 0; i < len; i++) {
		h ^= p[i];
		h *= 1099511628211ULL;
	}
	return h;
}

uint64_t arc_dsl_state_hash(const struct arc_game *game)
{
	const struct arc_dsl_spec *s = spec_of(game);
	const struct arc_dsl_aux *aux = (const struct arc_dsl_aux *)game->aux;
	size_t n = (size_t)s->grid_w * (size_t)s->grid_h;
	uint64_t h = 1469598103934665603ULL;

	h = fnv(h, &game->engine.level_index, sizeof(int32_t));
	h = fnv(h, aux->grid, n);
	h = fnv(h, aux->floor, n);
	h = fnv(h, &aux->player_x, sizeof(int32_t));
	h = fnv(h, &aux->player_y, sizeof(int32_t));
	return h;
}

static void load_level(struct arc_game *game)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	int32_t level = game->engine.level_index;
	int32_t n = s->grid_w * s->grid_h;
	const int8_t *src = s->layout + (size_t)level * n;

	memcpy(aux->grid, src, (size_t)n);
	memcpy(aux->floor, s->floor + (size_t)level * n, (size_t)n);
	aux->player_x = -1;
	aux->player_y = -1;
	for (int32_t y = 0; y < s->grid_h; y++) {
		for (int32_t x = 0; x < s->grid_w; x++) {
			if (aux->grid[idx(s, x, y)] == (int8_t)s->player_kind) {
				aux->player_x = x;
				aux->player_y = y;
			}
		}
	}
	aux->settled = 0;
	aux->steps = s->budget ? s->budget[level] : 0;
	aux->sel_x = -1;
	aux->sel_y = -1;
}

static int won(const struct arc_game *game)
{
	const struct arc_dsl_spec *s = spec_of(game);
	const struct arc_dsl_aux *aux = (const struct arc_dsl_aux *)game->aux;
	int32_t n = s->grid_w * s->grid_h;

	if (s->win_mode == ARC_DSL_WIN_NONE_LEFT) {
		for (int32_t i = 0; i < n; i++)
			if (aux->grid[i] == (int8_t)s->win_a)
				return 0;
		return 1;
	}
	if (s->win_mode == ARC_DSL_WIN_ALL_ON) {
		for (int32_t i = 0; i < n; i++)
			if (aux->floor[i] == (int8_t)s->win_b &&
			    aux->grid[i] != (int8_t)s->win_a)
				return 0;
		return 1;
	}
	if (s->win_mode == ARC_DSL_WIN_REACH)
		return aux->player_x >= 0 &&
		       aux->floor[idx(s, aux->player_x, aux->player_y)] ==
			       (int8_t)s->win_a;
	if (s->win_mode == ARC_DSL_WIN_MATCH) {
		/* Compared by colour, so the target can use its own kinds
		 * (unclickable) that look like the canvas kinds. */
		for (int32_t v = 0; v < s->match_h; v++)
			for (int32_t u = 0; u < s->match_w; u++) {
				int8_t a = aux->grid[idx(s, s->match_x0 + u, s->match_y0 + v)];
				int8_t b = aux->grid[idx(s, s->match_x1 + u, s->match_y1 + v)];
				int8_t ca = a < 0 ? s->background : s->kinds[a].color;
				int8_t cb = b < 0 ? s->background : s->kinds[b].color;

				if (ca != cb)
					return 0;
			}
		return 1;
	}
	return 0;
}

/* Advance one cell through the cycle a..b, if it is in it. */
static void cycle_cell(const struct arc_dsl_spec *s, struct arc_dsl_aux *aux,
		       int32_t x, int32_t y, int8_t a, int8_t b)
{
	int8_t k;

	if (!inside(s, x, y))
		return;
	k = aux->grid[idx(s, x, y)];
	if (k < a || k > b)
		return;
	aux->grid[idx(s, x, y)] = k == b ? a : (int8_t)(k + 1);
}

static void apply(struct arc_game *game, int32_t cx, int32_t cy, uint8_t effect,
		  int8_t a, int8_t b, int *blocked)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	int32_t n = s->grid_w * s->grid_h;

	switch (effect) {
	case ARC_DSL_BLOCK:
		*blocked = 1;
		break;
	case ARC_DSL_REMOVE:
		aux->grid[idx(s, cx, cy)] = ARC_DSL_EMPTY;
		break;
	case ARC_DSL_BECOME:
		aux->grid[idx(s, cx, cy)] = a;
		break;
	case ARC_DSL_TOGGLE:
		for (int32_t i = 0; i < n; i++) {
			if (aux->grid[i] == a)
				aux->grid[i] = b;
			else if (aux->grid[i] == b)
				aux->grid[i] = a;
		}
		break;
	case ARC_DSL_CYCLE: {
		int8_t here = aux->grid[idx(s, cx, cy)];
		uint8_t stencil = here >= 0 ? s->kinds[here].stencil : 0;

		cycle_cell(s, aux, cx, cy, a, b);
		if (stencil >= ARC_DSL_STENCIL_CROSS) {
			cycle_cell(s, aux, cx - 1, cy, a, b);
			cycle_cell(s, aux, cx + 1, cy, a, b);
			cycle_cell(s, aux, cx, cy - 1, a, b);
			cycle_cell(s, aux, cx, cy + 1, a, b);
		}
		if (stencil == ARC_DSL_STENCIL_BLOCK) {
			cycle_cell(s, aux, cx - 1, cy - 1, a, b);
			cycle_cell(s, aux, cx + 1, cy - 1, a, b);
			cycle_cell(s, aux, cx - 1, cy + 1, a, b);
			cycle_cell(s, aux, cx + 1, cy + 1, a, b);
		}
		break;
	}
	case ARC_DSL_WIN:
		arc_game_next_level(game);
		break;
	case ARC_DSL_LOSE:
		arc_game_lose(game);
		break;
	default:
		break;
	}
}

static int32_t count_kind(const struct arc_dsl_spec *s,
			  const struct arc_dsl_aux *aux, int8_t kind)
{
	int32_t n = s->grid_w * s->grid_h;
	int32_t total = 0;

	for (int32_t i = 0; i < n; i++)
		if (aux->grid[i] == kind)
			total++;
	return total;
}

static int predicate_holds(const struct arc_game *game,
			   const struct arc_dsl_rule *r, int32_t cx, int32_t cy)
{
	const struct arc_dsl_spec *s = spec_of(game);
	const struct arc_dsl_aux *aux = (const struct arc_dsl_aux *)game->aux;
	static const int32_t dx[4] = { 0, 0, -1, 1 };
	static const int32_t dy[4] = { -1, 1, 0, 0 };

	switch (r->predicate) {
	case ARC_DSL_IF_COUNT_LE:
		return count_kind(s, aux, r->pred_a) <= r->pred_b;
	case ARC_DSL_IF_NONE_LEFT:
		return count_kind(s, aux, r->pred_a) == 0;
	case ARC_DSL_IF_ADJACENT:
		for (int32_t i = 0; i < 4; i++) {
			int32_t nx = cx + dx[i];
			int32_t ny = cy + dy[i];

			if (inside(s, nx, ny) &&
			    aux->grid[idx(s, nx, ny)] == r->pred_a)
				return 1;
		}
		return 0;
	default:
		return 1;
	}
}

static void fire_rules(struct arc_game *game, uint8_t trigger, int8_t subject,
		       int32_t cx, int32_t cy, int *blocked)
{
	const struct arc_dsl_spec *s = spec_of(game);

	for (int32_t i = 0; i < s->num_rules; i++) {
		const struct arc_dsl_rule *r = &s->rules[i];

		if (!r->enabled || r->trigger != trigger)
			continue;
		if (r->subject >= 0 && r->subject != subject)
			continue;
		if (!predicate_holds(game, r, cx, cy))
			continue;
		apply(game, cx, cy, r->effect, r->effect_a, r->effect_b,
		      blocked);
	}
}

/* Move the selected object one cell; it is blocked by anything solid and
 * pushes nothing. Stepping onto a floor tile fires nothing: selection
 * games win by arrangement (ARC_DSL_WIN_ALL_ON / MATCH). */
static void move_selected(struct arc_game *game, int32_t dir)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	int32_t nx, ny;
	int8_t kind;

	if (aux->sel_x < 0)
		return;
	nx = aux->sel_x + DX[dir];
	ny = aux->sel_y + DY[dir];
	if (!inside(s, nx, ny) || aux->grid[idx(s, nx, ny)] != ARC_DSL_EMPTY)
		return;
	kind = aux->grid[idx(s, aux->sel_x, aux->sel_y)];
	aux->grid[idx(s, aux->sel_x, aux->sel_y)] = ARC_DSL_EMPTY;
	aux->grid[idx(s, nx, ny)] = kind;
	aux->sel_x = nx;
	aux->sel_y = ny;
}

/* Select the next selectable object in scan order after the current one. */
static void cycle_selection(struct arc_game *game)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	int32_t n = s->grid_w * s->grid_h;
	int32_t start = aux->sel_x < 0 ? -1 : idx(s, aux->sel_x, aux->sel_y);

	for (int32_t step = 1; step <= n; step++) {
		int32_t i = (start + step) % n;
		int8_t k = aux->grid[i];

		if (k >= 0 && s->kinds[k].selectable) {
			aux->sel_x = i % s->grid_w;
			aux->sel_y = i / s->grid_w;
			return;
		}
	}
}

static void try_move(struct arc_game *game, int32_t dir)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	int32_t nx, ny;
	int blocked = 0;
	int8_t target;

	if (s->control == ARC_DSL_CONTROL_SELECT) {
		move_selected(game, dir);
		return;
	}
	if (aux->player_x < 0)
		return;
	nx = aux->player_x + DX[dir];
	ny = aux->player_y + DY[dir];
	if (!inside(s, nx, ny))
		return;
	target = aux->grid[idx(s, nx, ny)];

	if (target != ARC_DSL_EMPTY) {
		const struct arc_dsl_kind *k = &s->kinds[target];

		if (k->on_enter == ARC_DSL_PUSH) {
			int32_t px = nx + DX[dir];
			int32_t py = ny + DY[dir];
			int8_t beyond;

			if (!inside(s, px, py))
				return;
			beyond = aux->grid[idx(s, px, py)];
			if (beyond != ARC_DSL_EMPTY)
				return;
			aux->grid[idx(s, px, py)] = target;
			aux->grid[idx(s, nx, ny)] = ARC_DSL_EMPTY;
		} else {
			apply(game, nx, ny, k->on_enter, k->enter_a, k->enter_b,
			      &blocked);
			fire_rules(game, ARC_DSL_ON_ENTER, target, nx, ny,
				   &blocked);
			if (blocked)
				return;
		}
	}

	aux->grid[idx(s, aux->player_x, aux->player_y)] = ARC_DSL_EMPTY;
	aux->player_x = nx;
	aux->player_y = ny;
	aux->grid[idx(s, nx, ny)] = (int8_t)s->player_kind;
}

static void do_click(struct arc_game *game, int32_t ax, int32_t ay)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	int32_t cx = (ax - s->origin_x) / s->pitch;
	int32_t cy = (ay - s->origin_y) / s->pitch;
	int blocked = 0;
	int8_t target;

	if (ax < s->origin_x || ay < s->origin_y || !inside(s, cx, cy))
		return;
	target = aux->grid[idx(s, cx, cy)];
	if (target == ARC_DSL_EMPTY)
		return;
	if (s->kinds[target].selectable) {
		aux->sel_x = cx;
		aux->sel_y = cy;
	}
	apply(game, cx, cy, s->kinds[target].on_click, s->kinds[target].click_a,
	      s->kinds[target].click_b, &blocked);
	fire_rules(game, ARC_DSL_ON_CLICK, target, cx, cy, &blocked);
}

static void step_actor(struct arc_game *game, int32_t x, int32_t y, int8_t kind,
		       uint8_t *moved)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	const struct arc_dsl_kind *k = &s->kinds[kind];
	int32_t dx = 0, dy = 0, nx, ny;
	int8_t target;

	if (k->motion == ARC_DSL_PATROL) {
		dx = DX[k->motion_a & 3];
		dy = DY[k->motion_a & 3];
	} else {
		int32_t gx = aux->player_x - x;
		int32_t gy = aux->player_y - y;

		int32_t ax, ay;

		if (k->motion == ARC_DSL_FLEE) {
			gx = -gx;
			gy = -gy;
		}
		ax = gx < 0 ? -gx : gx;
		ay = gy < 0 ? -gy : gy;
		if (ax >= ay && ax > 0)
			dx = gx > 0 ? 1 : -1;
		else if (ay > 0)
			dy = gy > 0 ? 1 : -1;
		if (dx == 0 && dy == 0)
			return;
	}
	nx = x + dx;
	ny = y + dy;
	if (!inside(s, nx, ny)) {
		if (k->motion == ARC_DSL_PATROL)
			aux->grid[idx(s, x, y)] = k->motion_b;
		return;
	}
	target = aux->grid[idx(s, nx, ny)];
	if (target == (int8_t)s->player_kind) {
		if (k->deadly)
			arc_game_lose(game);
		return;
	}
	if (target != ARC_DSL_EMPTY) {
		if (k->motion == ARC_DSL_PATROL)
			aux->grid[idx(s, x, y)] = k->motion_b;
		return;
	}
	aux->grid[idx(s, x, y)] = ARC_DSL_EMPTY;
	aux->grid[idx(s, nx, ny)] = kind;
	moved[idx(s, nx, ny)] = 1;
}

static void move_actors(struct arc_game *game)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	uint8_t moved[ARC_DSL_MAX_GRID * ARC_DSL_MAX_GRID];
	int32_t n = s->grid_w * s->grid_h;

	memset(moved, 0, (size_t)n);
	for (int32_t y = 0; y < s->grid_h; y++) {
		for (int32_t x = 0; x < s->grid_w; x++) {
			int8_t kind = aux->grid[idx(s, x, y)];

			if (kind < 0 || moved[idx(s, x, y)])
				continue;
			if (s->kinds[kind].motion == ARC_DSL_STATIC)
				continue;
			if (s->kinds[kind].motion != ARC_DSL_PATROL &&
			    aux->player_x < 0)
				continue;
			step_actor(game, x, y, kind, moved);
		}
	}
}

static int settle_once(struct arc_game *game)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	int moved = 0;

	for (int32_t y = s->grid_h - 2; y >= 0; y--) {
		for (int32_t x = 0; x < s->grid_w; x++) {
			int8_t kind = aux->grid[idx(s, x, y)];
			int8_t below;

			if (kind < 0 || !s->kinds[kind].gravity)
				continue;
			below = aux->grid[idx(s, x, y + 1)];
			if (below == (int8_t)s->player_kind) {
				if (s->kinds[kind].deadly)
					arc_game_lose(game);
				continue;
			}
			if (below != ARC_DSL_EMPTY)
				continue;
			aux->grid[idx(s, x, y)] = ARC_DSL_EMPTY;
			aux->grid[idx(s, x, y + 1)] = kind;
			moved = 1;
		}
	}
	return moved;
}

static void finish_action(struct arc_game *game)
{
	const struct arc_dsl_spec *s = spec_of(game);
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	struct arc_engine_state *e = &game->engine;

	if (e->status == NOT_FINISHED && won(game))
		arc_game_next_level(game);
	else if (e->status == NOT_FINISHED && s->budget &&
		 s->budget[e->level_index] > 0 && aux->steps <= 0)
		arc_game_lose(game);
	arc_game_complete_action(game);
}

static void dsl_on_set_level(struct arc_game *game)
{
	load_level(game);
}

static int s_uses5(const struct arc_game *game)
{
	return spec_of(game)->uses_action5;
}

static void dsl_step_once(struct arc_game *game)
{
	struct arc_engine_state *e = &game->engine;
	struct arc_dsl_aux *aux = (struct arc_dsl_aux *)game->aux;
	int32_t id = e->action_id;

	if (aux->phase) {
		if (settle_once(game) && aux->ticks < ARC_DSL_MAX_SETTLE) {
			aux->ticks++;
			return;
		}
		aux->phase = 0;
		finish_action(game);
		return;
	}

	if (id >= ARC_ACTION1 && id <= ARC_ACTION4)
		try_move(game, id - ARC_ACTION1);
	else if (id == ARC_ACTION5 && s_uses5(game))
		cycle_selection(game);
	else if (id == ARC_ACTION6)
		do_click(game, e->action_x, e->action_y);
	if (id != ARC_ACTION_RESET && aux->steps > 0)
		aux->steps--;

	if (e->status == NOT_FINISHED && id != ARC_ACTION_RESET)
		move_actors(game);
	{
		int blocked = 0;
		struct arc_dsl_aux *a = (struct arc_dsl_aux *)game->aux;

		fire_rules(game, ARC_DSL_ON_STEP, -1, a->player_x, a->player_y,
			   &blocked);
	}
	if (e->status == NOT_FINISHED && settle_once(game)) {
		aux->phase = 1;
		aux->ticks = 1;
		return;
	}
	finish_action(game);
}

static void paint(const struct arc_dsl_spec *s, int8_t *frame, int32_t x,
		  int32_t y, int8_t kind, int use_size)
{
	int8_t color = s->kinds[kind].color;
	int32_t span = s->pitch;
	int32_t x0, y0;

	if (use_size && s->kinds[kind].size > 0 &&
	    s->kinds[kind].size < s->pitch)
		span = s->kinds[kind].size;
	x0 = s->origin_x + x * s->pitch + (use_size ? s->kinds[kind].off_x : 0);
	y0 = s->origin_y + y * s->pitch + (use_size ? s->kinds[kind].off_y : 0);
	for (int32_t v = 0; v < span; v++) {
		int32_t row = y0 + v;

		if (row < 0 || row >= ARC_FRAME_SIZE)
			continue;
		for (int32_t u = 0; u < span; u++) {
			int32_t col = x0 + u;

			if (col < 0 || col >= ARC_FRAME_SIZE)
				continue;
			frame[row * ARC_FRAME_SIZE + col] = color;
		}
	}
}

static void paint_hud(const struct arc_dsl_spec *s,
		      const struct arc_dsl_aux *aux, int32_t level,
		      int8_t *frame)
{
	int32_t budget = s->budget ? s->budget[level] : 0;
	int32_t total, whole, rest, filled;
	int round_up;

	if (s->hud == ARC_DSL_HUD_NONE || budget <= 0)
		return;
	/* The same rounding the ported games use for their bars. */
	total = ARC_FRAME_SIZE * aux->steps;
	whole = total / budget;
	rest = total % budget;
	round_up = 2 * rest > budget || (2 * rest == budget && whole % 2 == 1);
	filled = whole + (round_up ? 1 : 0);
	if (filled < 0)
		filled = 0;
	if (filled > ARC_FRAME_SIZE)
		filled = ARC_FRAME_SIZE;
	for (int32_t i = 0; i < ARC_FRAME_SIZE; i++) {
		int8_t c = i < filled ? s->hud_on : s->hud_off;

		switch (s->hud) {
		case ARC_DSL_HUD_BOTTOM:
			frame[(ARC_FRAME_SIZE - 1) * ARC_FRAME_SIZE + i] = c;
			break;
		case ARC_DSL_HUD_TOP:
			frame[i] = c;
			break;
		case ARC_DSL_HUD_LEFT:
			frame[i * ARC_FRAME_SIZE] = c;
			break;
		case ARC_DSL_HUD_RIGHT:
			frame[i * ARC_FRAME_SIZE + ARC_FRAME_SIZE - 1] = c;
			break;
		default:
			break;
		}
	}
}

static void dsl_render_interface(struct arc_game *game, int8_t *frame)
{
	const struct arc_dsl_spec *s = spec_of(game);
	const struct arc_dsl_aux *aux = (const struct arc_dsl_aux *)game->aux;

	for (int32_t y = 0; y < s->grid_h; y++)
		for (int32_t x = 0; x < s->grid_w; x++) {
			int8_t floor = aux->floor[idx(s, x, y)];
			int8_t on_top = aux->grid[idx(s, x, y)];

			if (floor != ARC_DSL_EMPTY)
				paint(s, frame, x, y, floor, 0);
			else if (on_top != ARC_DSL_EMPTY)
				paint(s, frame, x, y, 0, 0);
		}
	for (int32_t y = 0; y < s->grid_h; y++)
		for (int32_t x = 0; x < s->grid_w; x++) {
			int8_t kind = aux->grid[idx(s, x, y)];

			if (kind != ARC_DSL_EMPTY)
				paint(s, frame, x, y, kind, 1);
		}
	if (aux->sel_x >= 0) {
		/* A one-pixel ring around the selected cell. */
		int32_t x0 = s->origin_x + aux->sel_x * s->pitch;
		int32_t y0 = s->origin_y + aux->sel_y * s->pitch;

		for (int32_t k = 0; k < s->pitch; k++) {
			int32_t xs[4] = { x0 + k, x0 + k, x0, x0 + s->pitch - 1 };
			int32_t ys[4] = { y0, y0 + s->pitch - 1, y0 + k, y0 + k };

			for (int32_t j = 0; j < 4; j++)
				if (xs[j] >= 0 && xs[j] < ARC_FRAME_SIZE &&
				    ys[j] >= 0 && ys[j] < ARC_FRAME_SIZE)
					frame[ys[j] * ARC_FRAME_SIZE + xs[j]] =
						s->select_color;
		}
	}
	paint_hud(s, aux, game->engine.level_index, frame);
}

const struct arc_hooks arc_dsl_hooks = { arc_dsl_zero_aux, dsl_on_set_level,
					 dsl_step_once, dsl_render_interface };
