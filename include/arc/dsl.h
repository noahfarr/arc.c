#ifndef ARC_DSL_H
#define ARC_DSL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "arc/game.h"

#define ARC_DSL_MAX_KINDS 16
#define ARC_DSL_MAX_LEVELS 12
#define ARC_DSL_MAX_GRID 32
#define ARC_DSL_EMPTY (-1)

enum {
	ARC_DSL_NONE = 0,
	ARC_DSL_BLOCK,
	ARC_DSL_REMOVE,
	ARC_DSL_PUSH,
	ARC_DSL_BECOME,
	ARC_DSL_TOGGLE,
	ARC_DSL_WIN,
	ARC_DSL_LOSE
};

enum { ARC_DSL_WIN_NONE_LEFT = 0, ARC_DSL_WIN_ALL_ON, ARC_DSL_WIN_REACH };

enum { ARC_DSL_ON_ENTER = 0, ARC_DSL_ON_CLICK, ARC_DSL_ON_STEP };

enum { ARC_DSL_STATIC = 0, ARC_DSL_CHASE, ARC_DSL_FLEE, ARC_DSL_PATROL };

enum {
	ARC_DSL_HUD_NONE = 0,
	ARC_DSL_HUD_BOTTOM,
	ARC_DSL_HUD_LEFT,
	ARC_DSL_HUD_TOP,
	ARC_DSL_HUD_RIGHT
};

enum {
	ARC_DSL_ALWAYS = 0,
	ARC_DSL_IF_COUNT_LE,
	ARC_DSL_IF_NONE_LEFT,
	ARC_DSL_IF_ADJACENT
};

#define ARC_DSL_MAX_RULES 8
#define ARC_DSL_MAX_SETTLE 24

struct arc_dsl_rule {
	uint8_t trigger;
	int8_t subject;
	uint8_t predicate;
	int8_t pred_a;
	int8_t pred_b;
	uint8_t effect;
	int8_t effect_a;
	int8_t effect_b;
	uint8_t enabled;
};

struct arc_dsl_kind {
	int8_t color;
	uint8_t motion;
	int8_t motion_a;
	int8_t motion_b;
	uint8_t deadly;
	uint8_t gravity;
	int8_t size;
	int8_t off_x;
	int8_t off_y;
	uint8_t on_enter;
	int8_t enter_a;
	int8_t enter_b;
	uint8_t on_click;
	int8_t click_a;
	int8_t click_b;
};

struct arc_dsl_spec {
	int32_t num_kinds;
	int32_t num_levels;
	int32_t grid_w;
	int32_t grid_h;
	int32_t pitch;
	int32_t origin_x;
	int32_t origin_y;
	int32_t player_kind;
	int32_t win_mode;
	int32_t win_a;
	int32_t win_b;
	int8_t background;
	int32_t hud;
	int8_t hud_on;
	int8_t hud_off;
	int32_t num_rules;
	struct arc_dsl_rule rules[ARC_DSL_MAX_RULES];
	struct arc_dsl_kind kinds[ARC_DSL_MAX_KINDS];
	const int8_t *layout;
	const int8_t *floor;
	const int32_t *budget;
};

struct arc_dsl_aux {
	int8_t grid[ARC_DSL_MAX_GRID * ARC_DSL_MAX_GRID];
	int8_t floor[ARC_DSL_MAX_GRID * ARC_DSL_MAX_GRID];
	int32_t player_x;
	int32_t player_y;
	uint8_t settled;
	uint8_t phase;
	int32_t ticks;
	int32_t steps;
};

extern const struct arc_hooks arc_dsl_hooks;

void arc_dsl_zero_aux(void *aux);
int32_t arc_dsl_num_actions(const struct arc_dsl_spec *spec);

#ifdef __cplusplus
}
#endif

#endif
