# Setting B adapters

This directory is the only tracked integration point for external COVID
baselines.  An adapter may call an isolated official repository through its
own environment, but must not copy or modify the upstream model core.

Each adapter must:

1. take a `COVIDSettingBProtocol` instance rather than loading the CDC data;
2. initialize from `protocol.task1()`;
3. call `protocol.week(t)` in chronological order and condition only on its
   delayed-hidden and current-visible observations;
4. write the prediction through `PredictionArchive`; and
5. retain the official-reproduction and protocol-audit JSON files beside its
   output archive.

The current hidden target is intentionally absent from `WeekInformation`.
