# What the generator is aiming at

This is the reference for the procedural generator: what the ARC-AGI-3 Kaggle
evaluation actually measures, what is publicly known about the hidden games,
what the 25 public games contain mechanic by mechanic, and where the DSL in
`src/dsl.c` stands against all of that. Every number below is sourced; the
primary source is the ARC Prize Foundation's technical report of 22 April 2026
(`ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence`, cited as
**[TR]**), the docs at docs.arcprize.org (**[docs]**), the Kaggle starter kit
(**[kit]**), and the shipped game metadata in `reference/games.json`
(**[ref]**).

## 1. The evaluation

### Sets

| Set | Environments | Use |
| --- | --- | --- |
| Public demo | 25 | Format demonstration. "Intentionally easier for both humans and AI", "does not comprehensively represent the mechanics found in the private set." |
| Semi-private | 55 | Frontier models behind an API. |
| Fully private | 55 | "Used for the competition. Only given to a very limited number of partners." |

The private sets are "intentionally out-of-distribution relative to the public
set. They cover a broader and more diverse set of mechanics with limited
overlap with the mechanics found in the public environments, and they probe
greater adaptation capabilities, involving deeper compositional reasoning."
[TR §3.6]

So the generator's target is not the 25 public games. The public games show
the *format* and the *design language*; the mechanics on the hidden set are
mostly different from them by construction.

### Format

- 64×64 frame, 16 colours, turn-based; the state never changes without an
  action. An action returns one frame or a frame sequence (animation). [TR §2.3]
- Action space per game is a declared subset of: ACTION1–4 (semantically
  up/down/left/right), ACTION5 ("interact, select, rotate, attach/detach,
  execute, etc."), ACTION6 (click at x,y in 0–63), ACTION7 (always undo where
  present), RESET. `available_actions` is in every frame's metadata; ACTION6
  availability never says *where* clicks are active. [docs/actions]
- Every environment has at least six levels; level 1 is a tutorial
  ("intentionally easy... random agents can occasionally stumble into success
  at this stage"). [TR §3.4]
- Frame metadata carries `levels_completed`, `win_levels`, `full_reset`,
  `available_actions`, `state ∈ {NOT_PLAYED, NOT_FINISHED, WIN, GAME_OVER}`.
- Competition mode (forced on Kaggle): one scorecard, one `make` per
  environment, every available environment is scored whether played or not,
  only level resets — a game reset becomes a level reset. [docs/competition_mode]

### Scoring: RHAE

Per level, with `h` the human baseline action count and `a` the agent's:

    S_level = min(1.15, (h / a)^2)

Per environment with `n` levels, level weight `w_l = l`, `k` levels completed:

    E = min( Σ_{l≤k} w_l / Σ w_l ,  Σ_l w_l · S_l / Σ w_l )

Total = mean of `E` over environments. [TR §4.1, docs/methodology]
On Kaggle the per-level cap is 1.0 instead of 1.15 (see below).

Consequences that matter for training:

- **Squared efficiency.** 2× the human action count is 25 %, 10× is 1 %.
  Exploration actions are charged at the same rate as execution actions.
- **Later levels dominate.** In a 6-level game, level 6 is worth 6/21 of the
  environment, level 1 is worth 1/21. Finishing 4 of 6 levels caps the
  environment at 10/21 ≈ 48 %.
- **Human baseline = upper-median best first-run human.** 10 members of the
  public per game; only environments fully solved by at least 2 of the 10 are
  included. Humans reset levels to improve efficiency and that counts against
  their action total, so the baseline already includes exploration. [TR §4.2, §5]
- **Action budget: 5× the human baseline per level**, after which the agent
  is terminated on that level. [TR §4.3]
- The foundation explicitly classifies "trained on many synthetically
  generated ARC-AGI-3 lookalike environments" as domain-specific overfitting
  that the *official* leaderboard discounts. [TR §4.3.1] This is irrelevant to
  the Kaggle track, which scores whatever is submitted, but it is the
  foundation's own statement that this approach can move the score.

### Kaggle track

Verified against the competition's own pages (fetched through Kaggle's API
on 2026-09-02; **[kaggle]**), which override the starter-kit summary:

- **110 hidden games.** "Competition evaluation uses a separate, private set
  of 110 games that your agent has never seen. Half of these are used for the
  Public Leaderboard score, and the other half for the Private Leaderboard
  score." So the public leaderboard is 55 games, the final ranking is the
  other 55. This matches the report's 55 semi-private + 55 fully private.
- **Per-level cap is 1.0 on Kaggle, not 1.15.** The data page states
  `min(human_actions / agent_actions, 1.0)` squared, and the evaluation page
  says "scores are capped at 100%". The 1.15 cap is the foundation's
  leaderboard, not the competition.
- **Runtime: 9 hours** per notebook, CPU or GPU (`maxCpuRuntimeMinutes =
  maxGpuRuntimeMinutes = 540`), synchronous rerun. Internet disabled.
  Submission size limit 20 GB. Accelerators: CPU, 2×T4, P100, RTX 6000
  (`g4-standard-48`, competition-only).
- **One submission per day** (`maxDailySubmissions = 1`), two selectable
  final submissions, scores shown to 2 decimals, team size ≤ 8.
- **External data and pretrained models are allowed** ("Freely & publicly
  available external data is allowed, including pre-trained models";
  rules §6: external data must be publicly available at no cost or meet the
  Reasonableness standard). Nothing in the rules restricts training on
  generated environments. Winners must open-source system, model and weights
  under CC-BY 4.0 and deliver training code (rules §5, §2.8).
- Agent interface: `is_done(frames, latest_frame)` and
  `choose_action(frames, latest_frame) -> GameAction`; the gateway records
  every action and writes `submission.parquet` (`row_id, game_id,
  end_of_game, score`); "as long as the agent takes action on *any* of the
  games, a submission file for all of the games is created." [kit, kaggle]
- Deadlines: milestones 30 June and 30 September 2026 (notebooks must be
  public to qualify), entry/team-merge 26 October, final 2 November, winners
  4 December 2026. Prizes: $75k final leaderboard, $75k milestones, $700k
  bonus for 100 %.
- **Still not published:** the exact action cap the gateway enforces per
  level (the report's 5× human is the foundation's own evaluation policy;
  the framework's default `MAX_ACTIONS` is 80 and must be raised by the
  agent) and any per-game wall-clock split inside the 9 h. Plan for
  110 games × (levels × 5×h) actions in 9 h: at the public-set median of
  638 human actions per game that is ~350k agent actions in 32,400 s, i.e.
  the whole pipeline needs to sustain ≥ 11 actions/s including inference.
- State of play on 2026-09-01: public leaderboard top is 7.51 %, tenth
  place 3.77 %, 2,706 teams. [kaggle]

### Human numbers on the public set

Per-level baselines for all 25 public games ship in `reference/games.json`
(these are the `h` values the toolkit uses). Aggregates:

| | value |
| --- | --- |
| levels per game | 6–10, median 7 (9 games have 6) |
| per-level human baseline | min 6, median 60, mean 94, p90 203, max 578 |
| level 1 | median 30, max 78 |
| last level | median 113 |
| whole game | median 638 actions, range 171–1843 |
| frames per action on the public set | ≈1.9 (37,658 frames over 19,772 actions in the parity harness) |

Design intent from [TR §3.4] worth treating as constraints on generated games:

- **Core Knowledge priors only**: objectness, basic geometry/topology
  (symmetry, rotation, inside/outside, connectedness, holes), basic physics
  (gravity, momentum, bouncing), agentness. No numbers, letters, real-world
  clip-art, or cultural colour conventions.
- **Novelty test**: two environments are "insufficiently distinct" if one
  program solves both while being ≥50 % shorter than the concatenation of two
  independent solvers.
- **Multiple mechanics per environment.** "Environments centered on a single
  mechanic that scaled in size or difficulty are treated as an anti-pattern."
- **Difficulty through composition**: later levels require integrating
  concepts learned earlier in the same environment.
- **Random-play bar**: a random policy must not win any non-tutorial level
  more than 1 in 10,000 times; validated with 50k and 1M-step random sweeps
  and a hash-merged state graph. (This is what `src/certify.c` and
  `harness/validate.py` reproduce.)
- Human-solvable in ~20 minutes; median successful attempt 8.1 minutes.

## 2. The 25 public games

Tags are from `reference/games.json`. `h` is the human baseline per level.
The mechanic descriptions come from reading `src/games/*.c`, which is the
verified port of the obfuscated reference.

| id | actions | levels | h (per level) | what it is |
| --- | --- | --- | --- | --- |
| ar25 | 1-4,5,6,7 | 8 | 32–233 | Mirror construction. Click (or ACTION5 to cycle) selects a movable shape or a mirror axis; arrows move it; the reflections of the shapes across the axes must cover a target pattern. Energy 64→320 per level. Undo. |
| bp35 | 3,4,6,7 | 9 | 21–163 | Side-view gravity platformer with vertical scroll. Walk left/right and fall; click tiles to break them, flip switches, arm/disarm bombs, grow "spread" tiles; a ceiling descends every 2 steps and crushes; spikes, gems, exit tile. 64/128 action cap. Undo. |
| cd82 | 1-4,5,6 | 6 | 8–55 | Painting. A tool sits at one of 8 positions around a 10×10 canvas; arrows move it; ACTION5 paints a position-dependent wedge in the current colour; click a swatch to change colour, click the arrow to paint a stripe. Match the answer picture. |
| cn04 | 1-4,5,6 | 6 | 29–300 | Jigsaw. Click selects a piece; arrows move it by one pixel; ACTION5 rotates it 90° (or cycles through alternative pieces of a group); win when every connector pixel is covered exactly twice. MaxSteps 75→200. |
| dc22 | 1-4,6 | 6 | 59–578 | Avatar walks on floor tiles to a goal; a crane driven by on-screen click buttons carries bridge pieces to lay new floor; tiles are revealed when stepped on; gated tiles, teleports, colour-cycling tiles; falling kills. Budget 128→1024. |
| ft09 | 6 | 6 | 12–65 | Lights-out generalisation. Clicking a cell applies a level-specific 3×3 stencil that advances the hit cells one step through a colour cycle; pattern cells carry their own stencil; win when every clue cell's neighbourhood equal/not-equal constraints hold. |
| g50t | 1-4,5 | 7 | 54–230 | Time echoes. Reach checkpoints in order; at each one your run ends, a new self spawns at the start, and the old self replays your moves as an echo that can hold buttons for gates and doors. Portals, sliding ice, a timer that ticks every 2 actions. ACTION5 rewinds a move. |
| ka59 | 1-4,6 | 7 | 28–326 | Container sokoban. Click selects a box; arrows move it, pushing chains of occupants; boxes must fit holes of exactly their size, markers must sit in zones; bombs tick a fuse per move and explode with recoil. Budget 100→200. |
| lf52 | 1-4,6,7 | 10 | 32–244 | Peg solitaire on irregular boards. Click a peg, then click a direction marker to jump over a neighbour and capture it; arrows bump the whole board (tiles shift, camera pans); ACTION5 reveals hidden images on pegs. Undo. One region uses unseeded randomness. |
| lp85 | 6 | 8 | 16–159 | Ring permutation. Clicking a button rotates the pieces along a closed path one step; paths overlap, so pieces transfer between rings. Every A piece on an A goal and every B piece on a B goal. Budget 13→150. |
| ls20 | 1-4 | 7 | 22–192 | Avatar with a loadout (shape, colour, rotation) that changes by stepping on buttons; must match the goal's spec and stand on it. Energy 42 per level with refill tiles, hazards, patrolling enemies, pushable blocks, fog on some levels, 3 lives. |
| m0r0 | 1-4,5,6 | 6 | 26–500 | Four movers in four quadrants driven by the same keys with mirrored axes; pairs must land on the same cell to merge. Standing on a switch opens same-coloured doors. Click a block to steer it instead. Hazards. |
| r11l | 6 | 6 | 22–52 | Drag-and-drop assembly. Click a piece, click a destination; fragments merge into composites on contact; a "key" is satisfied when its clue touches its target or a composite with exactly the right colour set; walls block, hazards knock back. |
| re86 | 1-4,5 | 8 | 26–424 | Multi-cursor construction. ACTION5 cycles which hollow rectangle is active; arrows move it; some cursors reshape (trade width for height) instead of moving; entering a flood tile recolours the cursor; the union of cursors must match a target image. Budget 100→400. |
| s5i5 | 6 | 8 | 20–162 | Articulated pipe trees. Click a coloured handle to rotate that subtree 90°; click a slider's half to lengthen/shorten; collisions revert; every endpoint on its target. Budget 50→200. |
| sb26 | 5,6,7 | 8 | 18–58 | Nested frames. Frames hold items, some of which are references to other frames; click to swap items or place from a tray; ACTION5 walks a cursor through the nesting and checks the flattened sequence against a card sequence. Energy. Undo. |
| sc25 | 1-4,6 | 6 | 6–143 | Spell drawing. Click cells in a 3×3 pad to draw a glyph; a permitted glyph casts teleport, 2× growth, or a fireball that clears blockers; avatar walks to the exit; icons replay a demo. |
| sk48 | 1-4,6,7 | 8 | 61–230 | Pistons on rails. Click a head; along its axis the arrows extend/retract it, pushing block chains; across the axis they slide it along a rail; the colour sequence of blocks in front of each head must match its markers. Budget 196. Undo. |
| sp80 | 1-4,5,6 | 6 | 25–152 | Liquid pouring. Click a block, arrows move it; ACTION5 pours from the spouts and the liquid flows through the level; it must reach the targets; four failed pours end the game. **Some levels rotate the whole scene and remap the arrow keys.** Steps 30→120. |
| su15 | 6,7 | 9 | 8–115 | Blob attraction. Clicking pulls nearby blobs toward the point; equal-tier blobs merge into the next tier; blobs swallow fruit; win when the counts of tiers/kinds inside the zones equal the level's targets. Steps 32/48. Undo. |
| tn36 | 6 | 7 | 26–72 | Programming. Toggle bits in instruction columns (opcodes: move, rotate, scale, recolour); click the switch to run the program on a robot that must end matching the target pose/colour without hitting walls or hazards; checkpoints; two lanes; preset buttons. |
| tr87 | 1-4 | 6 | 40–146 | Rewrite rules. A top row of glyphs, a bottom row, and displayed rules; left/right moves a cursor, up/down cycles the glyph under it; solved when the rules parse the top row into the bottom row. Later levels alter rules, use tree and double translation. |
| tu93 | 1-4 | 9 | 14–123 | Growing avatar. Consuming a crate of matching size grows you 3→5→7; push crates, drifting crates bounce off walls, "train" crates follow your path; reach the exit alive. StepCounter 20→60. |
| vc33 | 6 | 7 | 7–152 | Conservation. Levers move one unit of length from one pipe to another; sensors ride on the pipe ends; couplers fire a chained sequence of moves; every sensor must line up with a marker of its colour. StepCounter 50→200. |
| wa30 | 1-4,5 | 9 | 68–442 | Cooperative/adversarial box moving. ACTION5 grabs or releases the box you face; drag boxes to win tiles; allied seekers path-find boxes to targets on their own; thieves drag boxes to their own tiles and can be removed with ACTION5; holes. StepCounter 70→200. |

### Structural reading of the 25

**Control schemes** (this is the axis the DSL is weakest on):

| scheme | games |
| --- | --- |
| avatar + arrows | ls20, tu93, g50t (+5 rewind), wa30 (+5 grab), dc22 (+click crane), sc25 (+click pad), bp35 (left/right + click), m0r0 (four mirrored avatars + click) |
| click-to-select, arrows to move the selection | cn04, ka59, sk48, sp80, ar25, m0r0 (blocks) |
| ACTION5 as "cycle selection" | re86, ar25, cn04 (rotate) |
| ACTION5 as "execute / run / pour / paint / grab" | sb26, sp80, cd82, wa30 (tn36 does the same with a clicked switch) |
| cursor on a row, keys cycle the value | tr87 |
| click-only direct manipulation | ft09, lp85, r11l, s5i5, vc33, su15, tn36 |
| undo available | ar25, bp35, lf52, sb26, sk48, su15 (+ g50t's own rewind on 5) |

**Win conditions:**

| type | games |
| --- | --- |
| reach a goal cell (possibly with a state requirement) | ls20 (loadout), tu93, dc22, sc25, bp35, g50t (checkpoint chain) |
| all objects on matching targets | lp85, s5i5, vc33, ka59, wa30, r11l |
| merge/collect all | m0r0, lf52 (capture down), su15 (exact counts) |
| reproduce a target picture | cd82, re86, ar25, cn04 (pieces fit) |
| constraint satisfaction over a grid | ft09 |
| sequence / grammar match | tr87, sk48 (colour sequence), sb26 (flattened nesting), tn36 (program output) |

**Failure and pressure:** every game has a per-level step budget or energy
(shown as a bar on a frame edge, a shrinking timer, or a counter); several
add death (ls20 lives, tu93, dc22 falls, bp35 crush/spikes, g50t timer/death,
ka59 bombs, sp80 four failed pours). Budgets on the public set run about
1–3× the human baseline.

**Things the agent must discover, per game, with no instructions:** which
action ids do anything (13 of 25 games mix keys and click, and the click
region is never announced), what the avatar or selection is, what the goal
is, and what the level-to-level variation is. Hidden information appears
deliberately: fog (ls20), tiles revealed by stepping (dc22), hidden peg
images (lf52), rotated controls (sp80), hint/demo animations that replay a
solution fragment (ft09, ar25, sc25, su15 tutorial arrow).

**Autonomous agents and physics:** patrols (ls20), path-finding allies and
adversaries (wa30), drifting/bouncing and following objects (tu93), fuses and
timers (ka59, g50t, bp35), gravity and falling (bp35), attraction (su15),
flow (sp80), conservation of quantity (vc33), replay of your own past
(g50t), recursion (sb26).

**Frame statistics** (measured in `ad9906e`): real frames average 21
connected components of median size 9 px, ~7 colours, 30 % of the frame not
background. Every game draws a HUD: a budget bar along one edge and often a
level-progress strip.

## 3. Where `src/dsl.c` stands

What the DSL can express today: a tiled grid (≤32×32, ≤16 kinds, ≤8 rules),
one avatar moved by ACTION1–4, kinds with `on_enter` ∈ {block, remove, push,
become, toggle, win, lose} and `on_click`, rules with triggers
{enter, click, step} and conditions {always, count≤, none-left, adjacent},
actors {chase, flee, patrol} with `deadly`, gravity with a multi-frame
settle, win modes {none-left, all-on, reach}, per-object size/offset for
appearance, a level ladder. Every DSL game declares actions `[1,2,3,4,6]`.

Coverage against the public set, by the categories above:

| category | covered | not covered |
| --- | --- | --- |
| avatar + arrows | yes | avatar *state* (ls20 loadout, tu93 size, sc25 scale), lives, energy refill |
| click-to-select + arrows | no | cn04, ka59, sk48, sp80, ar25, m0r0 — 6 games |
| ACTION5 (any semantics) | no | 9 games declare it |
| ACTION7 undo | no | 6 games |
| click-only manipulation | partly (click toggles a tile) | rotation rings, rotate-subtree, sliders, drag-to-cell, attraction, program bits — 7 games |
| reach goal | yes | goal conditioned on state |
| all-on-targets | yes (push boxes) | target matching by size/colour/identity |
| picture / sequence / count wins | no | cd82, re86, ar25, cn04, tr87, sk48, sb26, tn36, ft09, su15 — 10 games |
| doors, switches, keys, collect | yes | switches that are *stood on* by a second actor (g50t echo, m0r0) |
| hazards, patrols, chase | yes | path-finding allies/adversaries, followers, bouncing drifters, fuses |
| gravity | yes (tiles fall) | side-view avatar physics, flow, attraction, conservation |
| hidden information | no | fog, reveal-on-step, hidden identity, rotated controls |
| multiple avatars / mirrored control | no | m0r0, tn36 lanes, re86 cursors |
| replay / recursion / rewrite | no | g50t, sb26, tr87 |
| HUD (budget bar, level strip) | no | all 25 |
| animation frames per action | settle only | ~1.9 frames/action on real games |

So the DSL reproduces the *avatar-on-tiles* corner of the public set well
(ls20, tu93 minus growth, the door-and-key half of dc22, the walking half of
sc25, wa30 minus the NPCs) and none of the other corners. Since the hidden
set has "limited overlap" with even the public mechanics, breadth of
primitives matters more than fidelity to any one public game.

## 4. What this implies for the generator

### 4.1 Where the score is

Level weights make the arithmetic simple. Solving *only* the tutorial level
of every hidden game at human efficiency is worth 1/21 of a 6-level game,
1/28 of a 7-level game, 1/36 of an 8-level game: about 4 % overall. Adding
level 2 at human efficiency roughly triples that. The public leaderboard top
on 2026-09-01 is 7.51 %, so the current frontier *is* "tutorial levels solved
efficiently, plus some level 2s". Every level beyond that is worth more than
the one before it, and the environment cap means nothing is earned for
efficiency on levels you never reach.

Squared efficiency turns exploration into a direct cost. On tu93 the BFS
optimum (measured with `harness.validate.explore` on the C port) against the
human baseline is:

| level | optimal | human | human/optimal |
| --- | --- | --- | --- |
| 1 | 18 | 19 | 1.06 |
| 2 | 10 | 16 | 1.6 |
| 3 | 19 | 34 | 1.8 |
| 4 | 17 | 42 | 2.5 |

At level 1 humans are already at the optimum: the goal is obvious from the
frame and they walk to it. An agent that spends 18 actions probing controls
before walking 18 gets 25 % of that level. Later levels leave a 1.5–2.5×
slack for exploration, and that is where humans spend theirs.

Two consequences for what we generate:

- **The tutorial-level prior is the single most valuable thing to get
  right**: what a first level looks like across control schemes, such that
  the goal and the controls can be read from one or two frames and executed
  in 10–30 actions with no probing.
- **Levels 2–n must reward carrying knowledge forward**, because that is
  where both the weight and the exploration slack are. "Difficulty through
  composition" is the in-context-learning curriculum, stated by the
  benchmark's designers.

### 4.2 Where the generator stood, and where it stands

Measured on 2026-09-02 before the fixes (`sample_environment` and
`sample_composed`, seed 0, BFS shortest path, 2,000-trial random play):

| family | level-1 optimal | later optimal | random win rate L1 / L2 / L3 |
| --- | --- | --- | --- |
| sokoban ladder | 1–6 | 7–20, unverified past L3 | 5–41 % / 3–19 % / 0.15–0.2 % |
| rooms (3 mechanics + hazard) | 15 | 21–22, several levels unverified | 16 % / 5 % / 2 % |
| public set (human) | median 30 | median 60, last level 113 | ≤ 1/10,000 for L2+ by construction |

Generated levels were 3–10× too short, tutorial levels were solved by a
random policy far too often, and every non-tutorial level failed the
foundation's 1/10,000 random-play bar by two to three orders of magnitude.
The DSL also had no step budget at all, so random play never ran out. An
agent trained on this learns that goals are adjacent and that flailing
works.

Same day, after the fixes (per-level budget with a HUD bar in the
interpreter; backward generation on reachable pulls biased away from the
goals with a far player start; both families as 6-level ladders whose
levels are redrawn until BFS puts the optimum inside a per-level band, with
budgets drawn at 2.5–4.5× the optimum; 10,000-trial random play):

| family | level-1 optimal | L2 / L3 optimal | random win rate L1 / L2+ |
| --- | --- | --- | --- |
| sokoban ladder | 16–24 | 20–25 / 24–28 | ≤ 4/10,000 / 0 |
| rooms ladder | 12–16 | 18–28 / 26–36 | ≤ 5/10,000 / ≤ 1/10,000 |

Levels 4–6 of both ladders are accepted on construction when BFS does not
finish within its node budget, so their optimum is unknown and their
budgets are heuristic; `harness.corpus.build` rejects any environment
whose non-tutorial levels a random policy wins more than 1 in 10,000 times.
Verifying solvability past level 3 without exhaustive BFS is still open.

### 4.3 Aligning the training signal with RHAE

The generator knows each level's optimum, which the benchmark does not
expose but which is the quantity the human baseline tracks. So the reward
for a generated level should be the metric itself with a surrogate baseline:

    r_level = min(1, (k_l · optimal_l / actions_l)^2) · l / Σ_l l

with `k_l` the human/optimal ratio (≈1 at level 1, 2–2.5 by level 4 on tu93;
this needs measuring on more of the public games), the cap at 1.0 because
Kaggle caps there, a level counted only when completed, and termination at
5× the surrogate baseline. Episode = the whole environment, levels in order,
level resets allowed and charged. This makes the environment cap, the level
weighting and the exploration cost all fall out of the reward rather than
being tuned separately.

**Hold the 25 public games out of training.** With one submission per day
the leaderboard cannot drive iteration; the public set is the only OOD
proxy for the hidden set (it is easier, so the proxy is optimistic, but the
direction is right). If the agent trains on them the proxy is gone.

### 4.4 Randomisation the hidden set forces

Things an agent cannot assume on a game it has never seen, so the generator
must vary them per game:

- which action ids exist (`available_actions`) and which do anything at
  all; the mapping of 1–4 to directions (sp80 rotates it), what 5 does,
  whether 7 exists;
- where clicks are meaningful (never announced) — a click head over the
  64×64 frame has to learn to find them from the picture;
- whether the controllable thing is an avatar, a selection, a cursor, or a
  global effect;
- appearance: colours, multi-pixel sprites (real median component is 9 px,
  not one tile), pitch and origin, HUD placement, animation frames (~1.9 per
  action on the public set);
- the goal type, and whether the goal is visible at all before interaction.

### 4.5 Throughput

110 games in 9 h with up to 5× human actions per level is ~350k actions in
32,400 s at the public-set median: the policy must sustain ≥ 11 actions/s
end to end on a T4 pair or an RTX 6000, so the in-context inference has to
be amortised in the network, not done by search at test time. The C
environment is not the bottleneck; the agent architecture is, and that
argues for training on many short generated levels rather than few long
ones.

### 4.6 Order of work

1. ~~Fix the distributions of the two existing families~~ (done
   2026-09-02, see §4.2; remaining: solvability past level 3 without BFS).
2. ~~Reward and protocol in relax-arc~~ (done 2026-09-02): `arc_vecenv`
   scores a completed level as `w_l · min(1, (baseline_l / actions_l)²)`
   and can lose a level at `cap × baseline_l` actions; public games use the
   human baselines from `reference/games.json`, generated games a
   `baselines` label of `k_l × optimum` with `k` from tu93/re86. In
   relax-arc, `environment=arc/generated` trains on a corpus directory and
   evaluates on the 23 public games held out, so `evaluation/episode_return`
   is a benchmark score. Open: `k_l` is measured on two games; unverified
   levels get a budget-derived baseline.
3. **New primitives in the tile interpreter**, in the order of §3: selection
   as a first-class object with ACTION5 semantics, win predicates
   (target picture, sequence, exact counts), avatar state and
   state-conditioned goals, HUD and budget in the frame, hidden information
   (fog, reveal-on-enter), path-finding actors, undo on a random subset of
   games, animation frames.
4. **A click-manipulation interpreter family** (rings, rotate-subtree,
   sliders, attraction/merge) — the largest uncovered block of the public
   set and the least shared code.
5. Measure `k_l` on more public games and re-measure frame statistics after
   each family lands, so fidelity is tracked rather than assumed.

Non-goals, from the design rules: no glyph alphabets, text or clip-art; no
environment built around one mechanic that only scales; generated families
should fail the foundation's own novelty test against each other (what
`harness/novelty.py` should measure).

## Sources

- ARC Prize Foundation, *ARC-AGI-3: A New Challenge for Frontier Agentic
  Intelligence*, 22 April 2026. https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf
- Scoring methodology: https://docs.arcprize.org/methodology
- Actions: https://docs.arcprize.org/actions
- Competition mode: https://docs.arcprize.org/toolkit/competition_mode
- Changelog (14 April 2026 scoring change, game versions): https://docs.arcprize.org/changelog
- Kaggle starter kit: https://docs.arcprize.org/arc-prize-2026 and https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter
- Competition page: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 (Description, Evaluation, Data, Code Requirements, Rules tabs) and https://arcprize.org/competitions/2026/arc-agi-3
- Per-level human baselines: `reference/games.json` (shipped with `arc-agi`)
- Per-level budgets: the `StepCounter` / `MaxSteps` / `steps` level data in `reference/*.py`
