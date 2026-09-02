#include "arc/vecenv.h"

#include <pthread.h>
#include <stdlib.h>

#define FRAME_BYTES (ARC_FRAME_SIZE * ARC_FRAME_SIZE)

struct worker_arg {
	struct arc_vec_env *vec;
	int32_t start;
	int32_t end;
	const int32_t *actions;
	int8_t *obs;
	float *reward;
	uint8_t *terminated;
	uint8_t *truncated;
	int32_t *level;
	int32_t *score;
	const uint8_t *restart_mask;
	struct slot *owner;
	int reset_only;
};

struct slot {
	struct arc_vec_env *vec;
	int32_t start;
	int32_t end;
	struct arc_render_scratch *scratch;
	int8_t frame[FRAME_BYTES];
};

struct arc_vec_env {
	struct arc_game **games;
	struct arc_game_spec *pool;
	int32_t num_games;
	int32_t *task;
	uint32_t *rng;
	int32_t num_envs;
	int32_t num_threads;
	int32_t horizon;
	int32_t *elapsed;
	int32_t *level_actions;
	int32_t reward_mode;
	float cap;
	pthread_t *threads;
	struct slot *slots;
	pthread_mutex_t lock;
	pthread_cond_t start_cv;
	pthread_cond_t done_cv;
	struct worker_arg proto;
	int32_t generation;
	int32_t pending;
	int packed;
	int stop;
};

static void emit(const struct worker_arg *a, int32_t i, struct arc_game *g)
{
	if (!a->vec->packed) {
		arc_game_frame(g, a->obs + (size_t)i * FRAME_BYTES);
		return;
	}
	int8_t *tmp = a->owner->frame;
	arc_game_frame(g, tmp);
	uint8_t *out = (uint8_t *)a->obs + (size_t)i * (FRAME_BYTES / 2);
	for (int32_t k = 0; k < FRAME_BYTES / 2; k++)
		out[k] = (uint8_t)((tmp[2 * k] & 0x0f) |
				   ((tmp[2 * k + 1] & 0x0f) << 4));
}

static int32_t draw(struct arc_vec_env *vec, int32_t i)
{
	uint32_t x = vec->rng[i];
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	vec->rng[i] = x;
	return (int32_t)(x % (uint32_t)vec->num_games);
}

static void assign(struct arc_vec_env *vec, int32_t i, int32_t k,
		   struct slot *owner)
{
	const struct arc_game_spec *s = &vec->pool[k];
	if (vec->games[i])
		arc_game_free(vec->games[i]);
	vec->games[i] = arc_game_new(s->levels, s->hooks,
				     (char *)s->aux_array +
					     (size_t)i * s->aux_stride,
				     s->statics, s->simple_actions,
				     s->num_simple, s->has_click,
				     s->max_frames);
	vec->task[i] = k;
	if (owner)
		arc_game_share_scratch(vec->games[i], &owner->scratch[k]);
}

static void restart(struct arc_vec_env *vec, int32_t i, struct slot *owner)
{
	if (vec->num_games > 1)
		assign(vec, i, draw(vec, i), owner);
	arc_game_init(vec->games[i]);
	vec->elapsed[i] = 0;
	vec->level_actions[i] = 0;
}

static float level_weight(const struct arc_game *g, int32_t level)
{
	int32_t n = g->levels->num_levels;
	return (float)(level + 1) / (float)(n * (n + 1) / 2);
}

/* Reward for the step that just happened, given the level before it and
 * the score delta; also applies the action cap. */
static float shape_reward(struct arc_vec_env *vec, int32_t i,
			  struct arc_game *g, int32_t before_level,
			  int32_t reward_i, uint8_t *term, uint8_t *capped)
{
	const int32_t *baseline = vec->pool[vec->task[i]].baseline;
	int32_t actions = ++vec->level_actions[i];

	if (vec->reward_mode == ARC_REWARD_LEVELS)
		return (float)reward_i;
	if (reward_i > 0) {
		float w = level_weight(g, before_level);
		float r = w;

		if (baseline && baseline[before_level] > 0) {
			float e = (float)baseline[before_level] / (float)actions;
			if (e > 1.0f)
				e = 1.0f;
			r = w * e * e;
		}
		vec->level_actions[i] = 0;
		return r;
	}
	if (vec->cap > 0.0f && baseline && baseline[before_level] > 0 &&
	    (float)actions >= vec->cap * (float)baseline[before_level] &&
	    !*term) {
		arc_game_lose(g);
		*term = 1;
		*capped = 1;
	}
	return 0.0f;
}

static void work(const struct worker_arg *a)
{
	struct arc_vec_env *vec = a->vec;
	for (int32_t i = a->start; i < a->end; i++) {
		struct arc_game *g = vec->games[i];
		if (a->reset_only) {
			restart(vec, i, a->owner);
			emit(a, i, vec->games[i]);
			continue;
		}
		int32_t reward_i;
		uint8_t term;
		int32_t before_level = g->engine.level_index;
		arc_game_step(g, a->actions[i], a->vec->packed ? a->owner->frame
							 : a->obs + (size_t)i * FRAME_BYTES,
			      &reward_i, &term);
		uint8_t capped = 0;
		float shaped = shape_reward(vec, i, g, before_level, reward_i,
					    &term, &capped);
		if (a->vec->packed) {
			uint8_t *out = (uint8_t *)a->obs +
				       (size_t)i * (FRAME_BYTES / 2);
			const int8_t *tmp = a->owner->frame;
			for (int32_t k = 0; k < FRAME_BYTES / 2; k++)
				out[k] = (uint8_t)((tmp[2 * k] & 0x0f) |
						   ((tmp[2 * k + 1] & 0x0f) << 4));
		}
		vec->elapsed[i] += 1;
		uint8_t trunc = vec->horizon > 0 &&
				vec->elapsed[i] >= vec->horizon && !term;
		a->reward[i] = shaped;
		a->terminated[i] = term;
		a->truncated[i] = trunc;
		if (a->level)
			a->level[i] = g->engine.level_index;
		if (a->score)
			a->score[i] = g->engine.score;
		if (a->restart_mask && a->restart_mask[i]) {
			term = 0;
			trunc = 1;
			a->terminated[i] = 0;
			a->truncated[i] = 1;
		}
		if (term && a->restart_mask && !trunc && !capped) {
			/* Inside a trial the task must not change: a game that
			 * ends is RESET as the benchmark does it (level reset
			 * after a loss, full reset after a win), and the pool
			 * is only redrawn at the trial boundary. A loss from
			 * the action cap is final, as it is in the benchmark. */
			arc_game_perform_action_frames(g, ARC_ACTION_RESET, 0, 0,
						       NULL, 0);
			vec->level_actions[i] = 0;
			emit(a, i, g);
			if (a->level)
				a->level[i] = g->engine.level_index;
			if (a->score)
				a->score[i] = g->engine.score;
		} else if (term || trunc) {
			restart(vec, i, a->owner);
			emit(a, i, vec->games[i]);
			if (a->level)
				a->level[i] = vec->games[i]->engine.level_index;
			if (a->score)
				a->score[i] = vec->games[i]->engine.score;
		}
	}
}

static void *attend(void *raw)
{
	struct slot *slot = (struct slot *)raw;
	struct arc_vec_env *vec = slot->vec;
	int32_t seen = 0;
	for (;;) {
		pthread_mutex_lock(&vec->lock);
		while (vec->generation == seen && !vec->stop)
			pthread_cond_wait(&vec->start_cv, &vec->lock);
		if (vec->stop) {
			pthread_mutex_unlock(&vec->lock);
			return NULL;
		}
		seen = vec->generation;
		struct worker_arg a = vec->proto;
		pthread_mutex_unlock(&vec->lock);

		a.start = slot->start;
		a.end = slot->end;
		a.owner = slot;
		work(&a);

		pthread_mutex_lock(&vec->lock);
		if (--vec->pending == 0)
			pthread_cond_signal(&vec->done_cv);
		pthread_mutex_unlock(&vec->lock);
	}
}

static void run(struct arc_vec_env *vec, struct worker_arg proto)
{
	proto.vec = vec;
	if (vec->num_threads <= 1) {
		proto.start = 0;
		proto.end = vec->num_envs;
		proto.owner = &vec->slots[0];
		work(&proto);
		return;
	}

	pthread_mutex_lock(&vec->lock);
	vec->proto = proto;
	vec->pending = vec->num_threads - 1;
	vec->generation++;
	pthread_cond_broadcast(&vec->start_cv);
	pthread_mutex_unlock(&vec->lock);

	struct slot *mine = &vec->slots[vec->num_threads - 1];
	proto.start = mine->start;
	proto.end = mine->end;
	proto.owner = mine;
	work(&proto);

	pthread_mutex_lock(&vec->lock);
	while (vec->pending > 0)
		pthread_cond_wait(&vec->done_cv, &vec->lock);
	pthread_mutex_unlock(&vec->lock);
}

struct arc_vec_env *arc_vecenv_new_pool(const struct arc_game_spec *pool,
					int32_t num_games, int32_t num_envs,
					int32_t num_threads, uint64_t seed)
{
	struct arc_vec_env *vec = calloc(1, sizeof(struct arc_vec_env));
	vec->num_envs = num_envs;
	vec->horizon = 0;
	vec->num_games = num_games;
	vec->pool = calloc(num_games, sizeof(struct arc_game_spec));
	for (int32_t k = 0; k < num_games; k++)
		vec->pool[k] = pool[k];
	vec->games = calloc(num_envs, sizeof(struct arc_game *));
	vec->task = calloc(num_envs, sizeof(int32_t));
	vec->rng = calloc(num_envs, sizeof(uint32_t));
	vec->elapsed = calloc(num_envs, sizeof(int32_t));
	vec->level_actions = calloc(num_envs, sizeof(int32_t));
	vec->reward_mode = ARC_REWARD_LEVELS;
	vec->cap = 0.0f;
	for (int32_t i = 0; i < num_envs; i++) {
		uint32_t state = (uint32_t)(seed + 0x9e3779b9u * (uint32_t)i);
		vec->rng[i] = state ? state : 1u;
		assign(vec, i, num_games > 1 ? draw(vec, i) : 0, NULL);
	}

	int32_t nt = num_threads < 1 ? 1 : num_threads;
	int32_t chunk = (num_envs + nt - 1) / nt;
	if (chunk < 1)
		chunk = 1;
	vec->slots = calloc(nt, sizeof(struct slot));
	int32_t used = 0;
	for (int32_t t = 0; t < nt; t++) {
		int32_t start = t * chunk;
		if (start >= num_envs)
			break;
		int32_t end = start + chunk;
		if (end > num_envs)
			end = num_envs;
		vec->slots[used].vec = vec;
		vec->slots[used].start = start;
		vec->slots[used].end = end;
		used++;
	}
	vec->num_threads = used;

	for (int32_t t = 0; t < used; t++) {
		vec->slots[t].scratch =
			calloc(num_games, sizeof(struct arc_render_scratch));
		for (int32_t k = 0; k < num_games; k++) {
			const struct arc_level_data *d = vec->pool[k].levels;
			arc_render_scratch_init_dims(&vec->slots[t].scratch[k],
						     d->ph, d->pw,
						     d->num_slots);
		}
		for (int32_t i = vec->slots[t].start; i < vec->slots[t].end;
		     i++)
			arc_game_share_scratch(
				vec->games[i],
				&vec->slots[t].scratch[vec->task[i]]);
	}

	if (used > 1) {
		pthread_mutex_init(&vec->lock, NULL);
		pthread_cond_init(&vec->start_cv, NULL);
		pthread_cond_init(&vec->done_cv, NULL);
		vec->threads = calloc(used - 1, sizeof(pthread_t));
		for (int32_t t = 0; t < used - 1; t++)
			pthread_create(&vec->threads[t], NULL, attend,
				       &vec->slots[t]);
	}
	return vec;
}

void arc_vecenv_free(struct arc_vec_env *vec)
{
	if (vec->num_threads > 1) {
		pthread_mutex_lock(&vec->lock);
		vec->stop = 1;
		pthread_cond_broadcast(&vec->start_cv);
		pthread_mutex_unlock(&vec->lock);
		for (int32_t t = 0; t < vec->num_threads - 1; t++)
			pthread_join(vec->threads[t], NULL);
		free(vec->threads);
		pthread_cond_destroy(&vec->done_cv);
		pthread_cond_destroy(&vec->start_cv);
		pthread_mutex_destroy(&vec->lock);
	}
	for (int32_t i = 0; i < vec->num_envs; i++)
		arc_game_free(vec->games[i]);
	for (int32_t t = 0; t < vec->num_threads; t++) {
		for (int32_t k = 0; k < vec->num_games; k++)
			arc_render_scratch_free(&vec->slots[t].scratch[k]);
		free(vec->slots[t].scratch);
	}
	free(vec->slots);
	free(vec->games);
	free(vec->pool);
	free(vec->task);
	free(vec->rng);
	free(vec->elapsed);
	free(vec->level_actions);
	free(vec);
}

int32_t arc_vecenv_num_actions(const struct arc_vec_env *vec)
{
	int32_t most = 0;
	for (int32_t i = 0; i < vec->num_envs; i++)
		if (vec->games[i]->num_actions > most)
			most = vec->games[i]->num_actions;
	return most;
}

void arc_vecenv_set_packed(struct arc_vec_env *vec, int32_t packed)
{
	vec->packed = packed ? 1 : 0;
}

void arc_vecenv_set_reward(struct arc_vec_env *vec, int32_t mode, float cap)
{
	vec->reward_mode = mode;
	vec->cap = cap;
}

void arc_vecenv_tasks(const struct arc_vec_env *vec, int32_t *out)
{
	for (int32_t i = 0; i < vec->num_envs; i++)
		out[i] = vec->task[i];
}

void arc_vecenv_action_ids(const struct arc_vec_env *vec, int32_t *out)
{
	for (int32_t i = 0; i < vec->num_envs; i++) {
		const struct arc_game *g = vec->games[i];
		int32_t bits = 0;
		for (int32_t k = 0; k < g->num_simple; k++)
			bits |= 1 << g->simple_actions[k];
		if (g->has_click)
			bits |= 1 << ARC_ACTION6;
		out[i] = bits;
	}
}

void arc_vecenv_action_counts(const struct arc_vec_env *vec, int32_t *out)
{
	for (int32_t i = 0; i < vec->num_envs; i++)
		out[i] = vec->games[i]->num_actions;
}

struct arc_vec_env *arc_vecenv_new(const struct arc_level_data *levels,
				   const struct arc_hooks *hooks,
				   void *aux_array, size_t aux_stride,
				   void *statics, const int32_t *simple_actions,
				   int32_t num_simple, int32_t has_click,
				   int32_t max_frames, int32_t num_envs,
				   int32_t num_threads)
{
	struct arc_game_spec spec = { levels,         hooks,
				      aux_array,      aux_stride,
				      statics,        simple_actions,
				      num_simple,     has_click,
				      max_frames,     NULL };
	return arc_vecenv_new_pool(&spec, 1, num_envs, num_threads, 1);
}

void arc_vecenv_reset(struct arc_vec_env *vec, int8_t *obs)
{
	struct worker_arg proto = { 0 };
	proto.vec = vec;
	proto.obs = obs;
	proto.reset_only = 1;
	run(vec, proto);
}

void arc_vecenv_step_trial(struct arc_vec_env *vec, const int32_t *actions,
			   const uint8_t *restart_mask, int8_t *obs,
			   float *reward, uint8_t *terminated,
			   uint8_t *truncated, int32_t *level, int32_t *score)
{
	struct worker_arg proto = { 0 };
	proto.vec = vec;
	proto.actions = actions;
	proto.restart_mask = restart_mask;
	proto.obs = obs;
	proto.reward = reward;
	proto.terminated = terminated;
	proto.truncated = truncated;
	proto.level = level;
	proto.score = score;
	run(vec, proto);
}

void arc_vecenv_step(struct arc_vec_env *vec, const int32_t *actions,
		     int8_t *obs, float *reward, uint8_t *terminated,
		     uint8_t *truncated, int32_t *level, int32_t *score)
{
	struct worker_arg proto = { 0 };
	proto.vec = vec;
	proto.actions = actions;
	proto.obs = obs;
	proto.reward = reward;
	proto.terminated = terminated;
	proto.truncated = truncated;
	proto.level = level;
	proto.score = score;
	run(vec, proto);
}
