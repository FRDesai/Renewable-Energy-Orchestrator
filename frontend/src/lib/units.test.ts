import { describe, expect, it } from "vitest";
import { mw, mwh, pct, usd, hz, tonnes, ms, signedMw, perMwh, deltaPct } from "./units";

describe("every figure on screen carries its unit", () => {
  it("formats power with MW", () => {
    expect(mw(80)).toBe("80 MW");
    expect(mw(44.78)).toBe("44.8 MW");
    expect(mw(0)).toBe("0 MW");
  });

  it("keeps the sign on a dispatch figure, because direction is the point", () => {
    expect(signedMw(80)).toBe("+80 MW");
    expect(signedMw(-44.8)).toBe("−44.8 MW");
    expect(signedMw(0)).toBe("0 MW");
  });

  it("uses a real minus sign, not a hyphen", () => {
    // A hyphen-minus reads as a dash at small sizes and breaks number alignment.
    expect(usd(-1200)).toContain("−");
    expect(usd(-1200)).not.toContain("-");
  });

  it("formats energy, percentage, frequency, carbon and latency", () => {
    expect(mwh(480)).toBe("480 MWh");
    expect(pct(31.25)).toBe("31%");
    expect(pct(31.25, 1)).toBe("31.3%"); // half-up, as an operator expects
    expect(hz(49.751)).toBe("49.75 Hz");
    expect(tonnes(506.4)).toBe("506 t CO₂");
    expect(ms(28.84)).toBe("29 ms");
  });

  it("formats money with thousands separators and no cents", () => {
    expect(usd(1183320)).toBe("$1,183,320");
    expect(usd(-15.4)).toBe("−$15");
  });

  it("formats a price as a rate, because $/MWh is not $", () => {
    expect(perMwh(17.16)).toBe("$17.16/MWh");
    expect(perMwh(-19.54)).toBe("−$19.54/MWh");
  });

  it("formats a delta with its direction and a percent sign", () => {
    expect(deltaPct(-6.0)).toBe("−6.0%");
    expect(deltaPct(34.4)).toBe("+34.4%");
    expect(deltaPct(0)).toBe("0.0%");
  });

  it("never renders NaN or Infinity at an operator", () => {
    for (const bad of [NaN, Infinity, -Infinity]) {
      expect(mw(bad)).toBe("—");
      expect(usd(bad)).toBe("—");
      expect(pct(bad)).toBe("—");
      expect(perMwh(bad)).toBe("—");
    }
  });
});
