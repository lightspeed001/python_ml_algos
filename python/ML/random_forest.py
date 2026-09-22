from sklearn.ensemble import RandomForestClassifier

# initialise 100 trees, each limited to depth 5
rf = RandomForestClassifier(n_estimator=100, # num of trees in forest
max_depth=5, # each tree depth capped for speed/interoperability
random_state=0)

# Train the forest on the same Iris data
rf.fit(X, y)
print("00B score: ", rf.oob_score)

# if hasattr(rf, "oon_score"):
# print("OOB score: ", rf.oob_score_)
# else:
# print("OOB score not computed (set oob_score=True to enable).")
