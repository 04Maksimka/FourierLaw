import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('data/resonance_analysis.txt', sep='\s+', skipfooter=1, engine='python')
df['Ratio'] = df['None_Trivial'] / df['Trivial']

scale_factor = df['Ratio'].iloc[0] * df['L'].iloc[0]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

# ВЕРХНИЙ ГРАФИК: Линейный масштаб
ax1.plot(df['L'], df['Ratio'], 'o-', label='Численный расчет')
ax1.plot(df['L'], scale_factor / df['L'], '--', color='gray', label=r'Аппроксимация $\sim 1/L$')

ax1.set_ylabel('Отношение Res$_{nontriv}$ / Res$_{triv}$', fontsize=12)
ax1.set_title('Доля нетривиальных резонансов по сравнению с тривиальными', fontsize=14)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.legend(fontsize=11, loc='upper right')

# НИЖНИЙ ГРАФИК: Логарифмический масштаб (log-log)
ax2.loglog(df['L'], df['Ratio'], 'o-', label='Численный расчет')
ax2.loglog(df['L'], scale_factor / df['L'], '--', color='gray', label=r'Аппроксимация $\sim 1/L$')


ax2.set_xlabel('Размер системы $L$', fontsize=12)
ax2.set_ylabel(r'$\log$(Res$_{nontriv}$ / Res$_{triv}$)', fontsize=12)
ax2.set_title('Логарифмический масштаб', fontsize=14)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.legend(fontsize=11, loc='upper right')

fig.tight_layout()


output_filename = 'resonance_ratio_combined_plot.png'
fig.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"График сохранен в файл: {output_filename}")

plt.show()