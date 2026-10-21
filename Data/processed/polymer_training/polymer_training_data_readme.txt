PolyLife Supervised Polymer Training Data
==========================================

UV_mass:
- Input condition: polymer material + UV exposure time
- Source measurement: mass change from baseline
- Target: normalized mass-loss degradation [0,1]

RSOS:
- Input condition: treatment + aging time
- Target: existing mechanical degradation index [0,1]

Important:
- Temperature is not present as a validated predictor in these tables.
- Pressure is not present as a validated predictor in these tables.
- Vibration is not present as a validated predictor in these tables.
- These variables must not be fabricated.

Validation:
Use temporal holdout / expanding-window validation within each trajectory.
Do not randomly split neighboring aging measurements.
