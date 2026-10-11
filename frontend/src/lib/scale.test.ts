import { describe, expect, it } from "vitest";
import { linear, niceTicks, stack, bandPath, linePath } from "./scale";

describe("linear scale", () => {
  it("maps a domain onto a range", () => {
    const x = linear([0, 10], [0, 100]);
    expect(x(0)).toBe(0);
    expect(x(5)).toBe(50);
    expect(x(10)).toBe(100);
  });

  it("inverts the range for a y axis, so larger values sit higher", () => {
    const y = linear([0, 100], [200, 0]);
    expect(y(0)).toBe(200);
    expect(y(100)).toBe(0);
  });

  it("survives a degenerate domain rather than dividing by zero", () => {
    const x = linear([5, 5], [0, 100]);
    expect(Number.isFinite(x(5))).toBe(true);
  });

  it("clamps nothing by default: an out-of-domain value extrapolates", () => {
    const x = linear([0, 10], [0, 100]);
    expect(x(20)).toBe(200);
  });
});

describe("nice ticks", () => {
  it("returns round numbers covering the domain", () => {
    const ticks = niceTicks(0, 330, 5);
    expect(ticks[0]).toBeLessThanOrEqual(0);
    expect(ticks[ticks.length - 1]).toBeGreaterThanOrEqual(330);
    for (const t of ticks) expect(Number.isFinite(t)).toBe(true);
  });

  it("includes zero when the domain spans it, so the baseline is a real tick", () => {
    expect(niceTicks(-30, 90, 5)).toContain(0);
  });

  it("handles a flat domain without looping forever", () => {
    const ticks = niceTicks(7, 7, 5);
    expect(ticks.length).toBeGreaterThan(0);
    expect(ticks.length).toBeLessThan(20);
  });
});

describe("stacking", () => {
  it("accumulates series into [y0, y1] pairs per point", () => {
    const layers = stack([
      [1, 2],
      [3, 4],
    ]);
    expect(layers[0]).toEqual([
      [0, 1],
      [0, 2],
    ]);
    expect(layers[1]).toEqual([
      [1, 4],
      [2, 6],
    ]);
  });

  it("treats a missing point as zero rather than breaking the stack", () => {
    const layers = stack([[1], [3, 4]]);
    expect(layers[1]?.[1]).toEqual([0, 4]);
  });
});

describe("paths", () => {
  const x = linear([0, 2], [0, 100]);
  const y = linear([0, 10], [50, 0]);

  it("draws a line through the points", () => {
    expect(linePath([0, 5, 10], x, y)).toBe("M0,50L50,25L100,0");
  });

  it("skips a null point so a gap stays a gap", () => {
    expect(linePath([0, null, 10], x, y)).toBe("M0,50M100,0");
  });

  it("closes a band by walking the lower edge back", () => {
    const d = bandPath([0, 2], [8, 10], x, y);
    expect(d.startsWith("M0,")).toBe(true);
    expect(d.endsWith("Z")).toBe(true);
  });

  it("returns nothing drawable for an empty series", () => {
    expect(linePath([], x, y)).toBe("");
    expect(bandPath([], [], x, y)).toBe("");
  });
});
