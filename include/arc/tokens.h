#ifndef ARC_TOKENS_H
#define ARC_TOKENS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* A frame as objects: every 4-connected region of one colour becomes a
 * token, largest first, so a policy can attend over objects instead of
 * pixel patches and click a region wherever its cells happen to lie. */
#define ARC_TOKEN_FIELDS 10

struct arc_token {
	int16_t color;
	int16_t cx, cy;     /* centroid, rounded */
	int16_t x0, y0;     /* bounding box, inclusive */
	int16_t x1, y1;
	int16_t area;       /* pixels; 0 marks padding */
	int16_t px, py;     /* a pixel of the region nearest its centroid */
};

/* Tokenise a 64x64 frame of palette indices into at most cap tokens,
 * sorted by area (descending, ties by first pixel in scan order) and
 * returns how many were written; the rest of out (up to cap) is zeroed.
 * labels, when given, receives per pixel the index of its token or -1
 * when the region was dropped for exceeding cap. */
int32_t arc_frame_tokens(const int8_t *frame, struct arc_token *out,
			 int32_t cap, int16_t *labels);

#ifdef __cplusplus
}
#endif

#endif
