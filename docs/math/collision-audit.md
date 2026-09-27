# Collision audit — game 1.7.0

The engine stores a horizontal position `x` as fixed-point units, with `U`
units per tile and 80 source pixels per tile. A zombie body is

```text
[80x, 80x + 42U]
```

and a plant body in column `c` is `[80cU + 10U, 80cU + 70U]`. A walking
zombie's ordinary attack rectangle is `[80x + 14U, 80x + 34U]`; a pole
carrier uses `[80x - 65U, 80x + 5U]`. Contact is eligible when

```text
max(0, min(rights) - max(lefts)) >= 20U.
```

The equality case is intentional: at the first ordinary contact the overlap is
20 source pixels. At the right boundary, shifting one fixed-point unit farther
right reduces the overlap by 80/U source pixels; at U=80 this leaves 19 pixels.
The ordinary valid interval relative to a plant is [-4,36] source pixels and
the pole interval is [25,115], including both endpoints.
The engine tests attack geometry separately from tile occupancy, so a
zombie may be on a plant's tile without acquiring a target.

Peas use the swept horizontal interval `[80x - 15U, 80x + 40U]`. Projectile
collision requires strictly positive rectangle overlap; equality is a miss.
Zombie and projectile movement both use swept intervals, which catches a
crossing between sampled endpoints. Circular bomb and mine tests retain

```text
dx² + dy² <= radius²
```

so circle tangency qualifies. Idle mower contact uses the swept zombie body and
the mower interval `[80m - 50U, 80m]` with strict positive overlap.

Walking resolves before bite acquisition. A normal zombie checks its bite age
every four ticks, or every eight ticks while chilled. Acquisition, release and
damage all use that cadence; biting freezes movement. Armed and rising mines
remain bite targets but take no ordinary bite damage; underground mines take
damage normally. Pole carriers check the
pre-movement attack rectangle, then keep a separate flight state. These rules
are mirrored in the CUDA kernel and are independent of policy inputs.

Independent controls cover both attack edges, every cadence residue, chilled
and headless states, pole approach/flight/removal, positive projectile overlap
versus inclusive blast tangency, lane isolation, and custom coordinate scales.
Zombie source velocity uses its own thousandths scale, independent of U.
Position/entity target tie-breaking, fixed-point rounding, custom waves and
the gameplay RNG are intentional approximations, not original-binary parity.
