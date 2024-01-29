# Heatmap Plot Example
sns.heatmap(df.resample('M').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
#plt.matshow(df.resample('M').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')  # Matshow not as good
plt.title('resampled over month', size=15)
#plt.colorbar()
plt.margins(0.02)
