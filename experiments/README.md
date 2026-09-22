# Parameter-grid experiments

20 dataset seeds x 5 learning rates x 5 epoch budgets. All 500 results are included, without filtering failures.

From the repository root:

~~~sh
python experiments/run.py --case 0
~~~

Compare the JSON with results/case-0000.json. Case identifiers range from 0 to 499.

This grid repeatedly evaluates held-out data for educational comparison. Do not treat its best score as an unbiased model-selection result; use a separate validation and final test split for that purpose.

Results were computed in September 2026. Commit timestamps follow the retrospective timeline explained in ../HISTORY.md.
