# %%
%load_ext autoreload
%autoreload 2

# %%
import torch

from chai_lab.interp.max_acts.max_acts_analyzer import MaxActsAnalyzer

ngrok_url = "https://ec18-2601-643-867e-39a0-d14a-9df3-80d8-7273.ngrok-free.app"

torch.set_grad_enabled(False)

# %%
analyzer = MaxActsAnalyzer(ngrok_url)

# %%
pdb_id = "1bbc"

# %%
analyzer.plot_pairwise_distance(pdb_id)

# %%
# Lvl 0 Mech Interp
analyzer.plot_acts_norm_for_pdb(pdb_id)

# %%
# Clean it up
analyzer.plot_pairwise_distance(pdb_id)
analyzer.plot_acts_norm_for_pdb(pdb_id, clip_max=800)

# %%
# On Diagonal Features
analyzer.plot_top_feature_at_location("1bbc", 80, 80, 1, plot_inclusion=True)


# %%
# Off Diagonal Features
analyzer.plot_top_feature_at_location("1bbc", 90, 91, 2, plot_inclusion=False)

# %%
# Close Pairs Feature
analyzer.plot_top_feature_at_location("1bbc", 90, 91, 0, plot_inclusion=False)

# %%
# Far Pairs Feature (try both 0 and 2)
analyzer.plot_top_feature_at_location("1bbc", 35, 78, 2, plot_inclusion=False)

# %%
# Excluding super close and super far?  Interesting!
analyzer.plot_top_feature_at_location("1bbc", 32, 48, 0, plot_inclusion=False)

# %%
# Let's get to the really interesting stuff!

# Alpha Helix Feature
analyzer.plot_top_feature_at_location("1bbc", 90, 91, 4, plot_inclusion=False)

# %%
# Beta Sheet Feature (Only loosely correlated? -- show if they want)
analyzer.plot_top_feature_at_location("11ba", 61, 62, 5, plot_inclusion=False)

# %%
# Disulfide Bond Feature
analyzer.plot_feature_vals("1bbc", 32543)

# %%
analyzer.plot_max_acts_table(32543, 0, 20)

# %%
# Cis-peptide bond feature
analyzer.plot_feature_vals("1a5q", 18180)

# %%
analyzer.plot_max_acts_table(18180, 0, 20)

# %%
# Also tracks locations that don't involve proline
analyzer.plot_max_acts_table(18180, 90, 100)
