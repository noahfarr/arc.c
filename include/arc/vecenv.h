#ifndef ARC_VECENV_H
#define ARC_VECENV_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#include "arc/game.h"

struct arc_vec_env;

struct arc_game_spec {
	const struct arc_level_data *levels;
	const struct arc_hooks *hooks;
	void *aux_array;
	size_t aux_stride;
	void *statics;
	const int32_t *simple_actions;
	int32_t num_simple;
	int32_t has_click;
	int32_t max_frames;
	/* Per-level baseline action counts (human, or k x optimal), or NULL. */
	const int32_t *baseline;
	/* Solver distance-to-win tables for potential-based shaping, or
	 * NULL: for level l the entries dist_offset[l] .. dist_offset[l+1]
	 * of dist_hash (sorted) and dist_val, keyed by state_hash(game). */
	uint64_t (*state_hash)(const struct arc_game *game);
	const uint64_t *dist_hash;
	const int32_t *dist_val;
	const int32_t *dist_offset;
};

enum { ARC_REWARD_LEVELS = 0, ARC_REWARD_RHAE = 1 };

struct arc_vec_env *arc_vecenv_new_pool(const struct arc_game_spec *pool,
					int32_t num_games, int32_t num_envs,
					int32_t num_threads, uint64_t seed);
void arc_vecenv_set_packed(struct arc_vec_env *vec, int32_t packed);
/* ARC_REWARD_LEVELS: +1 per level (the default). ARC_REWARD_RHAE: on
 * completing level l, l * min(1, (baseline_l / actions_l)^2): the
 * benchmark's per-level score in level units, so level 1 at human
 * efficiency is worth exactly 1 and later levels more, as the benchmark
 * weights them. Dividing by 1 + ... + n gives the benchmark's game score;
 * that is left to the caller so the learner sees the same unit in every
 * game. A game without a baseline falls back to +l. With cap > 0 a level
 * that runs past cap * baseline_l actions is lost, as the benchmark
 * terminates it. */
void arc_vecenv_set_reward(struct arc_vec_env *vec, int32_t mode, float cap);
/* Potential-based shaping from the distance tables: each step adds
 * weight * w_l * (phi(s') - phi(s)) with phi(s) = -dist(s) / dist(start),
 * so an optimal solve of level l collects exactly weight * w_l on the way,
 * the same as its completion is worth. States off the table (levels the
 * solver did not finish, or unreachable ones) get no shaping. */
void arc_vecenv_set_shaping(struct arc_vec_env *vec, float weight);
void arc_vecenv_tasks(const struct arc_vec_env *vec, int32_t *out);
void arc_vecenv_action_ids(const struct arc_vec_env *vec, int32_t *out);
void arc_vecenv_action_counts(const struct arc_vec_env *vec, int32_t *out);

struct arc_vec_env *arc_vecenv_new(const struct arc_level_data *levels,
				   const struct arc_hooks *hooks,
				   void *aux_array, size_t aux_stride,
				   void *statics, const int32_t *simple_actions,
				   int32_t num_simple, int32_t has_click,
				   int32_t max_frames, int32_t num_envs,
				   int32_t num_threads);
void arc_vecenv_free(struct arc_vec_env *vec);
int32_t arc_vecenv_num_actions(const struct arc_vec_env *vec);
void arc_vecenv_reset(struct arc_vec_env *vec, int8_t *obs);
void arc_vecenv_step_trial(struct arc_vec_env *vec,
			   const int32_t *actions,
			   const uint8_t *restart_mask, int8_t *obs,
			   float *reward, uint8_t *terminated,
			   uint8_t *truncated, int32_t *level, int32_t *score);
void arc_vecenv_step(struct arc_vec_env *vec, const int32_t *actions,
		     int8_t *obs, float *reward, uint8_t *terminated,
		     uint8_t *truncated, int32_t *level, int32_t *score);

#ifdef __cplusplus
}
#endif

#endif
