Round 2 of the cheap whole-field selection (2026-10-14): the analysis behind selection v4 (srl_search/field.py).
Scripts ran in this order in one Python session (rat kernel on search-dev/.venv), each using the previous ones' variables:
strata, evalset, evalfn, dec_ev, curve, fps, stud, al, al_train, sim1, sim2, sim3, thr, miss, miss2, cite, stack,
stack2, stack3, dry, eval4. The candidates of selection v3 were rebuilt from its saved teacher answers (no cost).
eval_population.json (one level up): Kimi K3 labels of 1,585 papers sampled in strata from v3's 690,564 candidates,
with each stratum's size, so precision and recall can be estimated for the whole population.
