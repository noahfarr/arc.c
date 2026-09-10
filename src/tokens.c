#include "arc/tokens.h"

#include <string.h>

#include "arc/engine.h"

#define N (ARC_FRAME_SIZE * ARC_FRAME_SIZE)

struct region {
	int32_t first;
	int32_t area;
	int32_t sx, sy;
	int16_t x0, y0, x1, y1;
	int8_t color;
};

static int32_t find(int32_t *parent, int32_t i)
{
	while (parent[i] != i) {
		parent[i] = parent[parent[i]];
		i = parent[i];
	}
	return i;
}

static void unite(int32_t *parent, int32_t a, int32_t b)
{
	a = find(parent, a);
	b = find(parent, b);
	if (a == b)
		return;
	/* Keep the smaller index as root so roots are the first pixels. */
	if (a < b)
		parent[b] = a;
	else
		parent[a] = b;
}

int32_t arc_frame_tokens(const int8_t *frame, struct arc_token *out,
			 int32_t cap, int16_t *labels)
{
	static __thread int32_t parent[N];
	static __thread int32_t root_index[N];
	static __thread struct region regions[N];
	static __thread int32_t order[N];
	int32_t num_regions = 0;

	for (int32_t i = 0; i < N; i++)
		parent[i] = i;
	for (int32_t y = 0; y < ARC_FRAME_SIZE; y++)
		for (int32_t x = 0; x < ARC_FRAME_SIZE; x++) {
			int32_t i = y * ARC_FRAME_SIZE + x;
			if (x > 0 && frame[i - 1] == frame[i])
				unite(parent, i, i - 1);
			if (y > 0 && frame[i - ARC_FRAME_SIZE] == frame[i])
				unite(parent, i, i - ARC_FRAME_SIZE);
		}
	for (int32_t i = 0; i < N; i++) {
		int32_t r = find(parent, i);
		int32_t x = i % ARC_FRAME_SIZE, y = i / ARC_FRAME_SIZE;
		struct region *reg;
		if (r == i) {
			root_index[i] = num_regions;
			reg = &regions[num_regions++];
			reg->first = i;
			reg->area = 0;
			reg->sx = reg->sy = 0;
			reg->x0 = reg->x1 = (int16_t)x;
			reg->y0 = reg->y1 = (int16_t)y;
			reg->color = frame[i];
		} else {
			reg = &regions[root_index[r]];
		}
		reg->area++;
		reg->sx += x;
		reg->sy += y;
		if (x < reg->x0)
			reg->x0 = (int16_t)x;
		if (x > reg->x1)
			reg->x1 = (int16_t)x;
		if (y < reg->y0)
			reg->y0 = (int16_t)y;
		if (y > reg->y1)
			reg->y1 = (int16_t)y;
	}
	/* Sort region indices by area descending, first pixel ascending. */
	for (int32_t i = 0; i < num_regions; i++)
		order[i] = i;
	for (int32_t i = 1; i < num_regions; i++) {
		int32_t v = order[i], j = i;
		while (j > 0 && (regions[order[j - 1]].area < regions[v].area ||
				 (regions[order[j - 1]].area == regions[v].area &&
				  regions[order[j - 1]].first > regions[v].first))) {
			order[j] = order[j - 1];
			j--;
		}
		order[j] = v;
	}
	int32_t count = num_regions < cap ? num_regions : cap;
	/* rank[region] = token index or -1 */
	static __thread int32_t rank[N];
	for (int32_t i = 0; i < num_regions; i++)
		rank[i] = -1;
	memset(out, 0, (size_t)cap * sizeof *out);
	for (int32_t t = 0; t < count; t++) {
		const struct region *reg = &regions[order[t]];
		rank[order[t]] = t;
		out[t].color = reg->color;
		out[t].cx = (int16_t)((2 * reg->sx + reg->area) / (2 * reg->area));
		out[t].cy = (int16_t)((2 * reg->sy + reg->area) / (2 * reg->area));
		out[t].x0 = reg->x0;
		out[t].y0 = reg->y0;
		out[t].x1 = reg->x1;
		out[t].y1 = reg->y1;
		out[t].area = (int16_t)reg->area;
		out[t].px = -1;
		out[t].py = -1;
	}
	/* Click pixel: the region's pixel nearest its centroid. */
	static __thread int32_t best[N];
	for (int32_t t = 0; t < count; t++)
		best[t] = 1 << 30;
	for (int32_t i = 0; i < N; i++) {
		int32_t t = rank[root_index[find(parent, i)]];
		if (labels)
			labels[i] = (int16_t)t;
		if (t < 0)
			continue;
		int32_t x = i % ARC_FRAME_SIZE, y = i / ARC_FRAME_SIZE;
		int32_t dx = x - out[t].cx, dy = y - out[t].cy;
		int32_t d = dx * dx + dy * dy;
		if (d < best[t]) {
			best[t] = d;
			out[t].px = (int16_t)x;
			out[t].py = (int16_t)y;
		}
	}
	return count;
}
