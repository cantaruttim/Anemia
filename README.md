# Anemia
 
Neste projeto, realizo a análise de um dataset que serve para classificar se um paciente pode ou não ter anemia, por meio dos parâmetros analisados para um Eritograma. A análise de dados focada na área da saúde pode contribuir positivamente para que pacientes sejam direcionados de forma a otimizar a sua terapia e a diminuir quadros sintomáticos secundários, diminuindo assim a chance de desenvolver novos quadros clínicos, que podem ser por vezes, mais complexos do que o atual.

1. O Eritograma que analisamos hoje, é composto por: 
* Hemoglobina (`Hemoglobin`)
* Concentração corpucular Média (`MCH`)
* Concentração de Hemoglobina corpuscular média (`MCHC`)
* Volume Corpuscular Médio (`MCV`)

### Importando as bibliotecas
```
import pandas as pd
import seaborn as sns
import numpy as n
import matplotlib.pyplot as plt
%matplotlib inline
```

Esses valores estão dispostos como segue:

- Representação do dataset utilizado no projeto

| Dataset        | Gender       | Hemoglobin  | MCH  | MCHC | MCV | Result |
| ------------- |:-------------:| -----------:|-----:|-----:|----:|-------:|
| Index ... | 1 | 14.9 | 22.7 | 29.1 | 83.7 | 0 |

Para o  `Gender` : 0 Masculino & 1 Feminino <br />
Para o `Result` : 0 Não anêmico & 1 Anêmico

### Análise Exploratória dos Dados

Nesta parte podemos observar a relação entre as variáveis e como elas se comportam em relação as diferentes nuances dos dados.
É muito interessante quando plotamos o resultado da correlação em um mapa de calor

`sns.heatmap(df.corr(), annot=True,linewidths=.5)`
_Por meio deste gráfico observamos que não há correlações plausíveis para que possamos inferir nenhum tipo de análise. Somente que os dados levam à resultados de anêmicos e de não anêmicos. Portanto entramos em um caso de classificação e não de regressão linear. Mesmo a correlação mais fraca está apontando que o resultado sofre algum tipo de influência por causa do gênero da pessoa. Porem basear os resultados com base nessa informação, é muito ruim_

![image](https://user-images.githubusercontent.com/81988636/207465951-6e3e1796-e1ff-4470-a197-78da65f1f854.png)

1. Porem podemos observar as variáveis de forma individual para entendermos como elas se comportam. E a que mais me chamou a atenção foi a da própria Hemoglobina

`sns.histplot(df['Hemoglobin'], kde=True);`

![image](https://user-images.githubusercontent.com/81988636/207466462-fa09326f-a934-4823-9215-87316ab37fb2.png) <br />
_Encontramos valores que não seguem uma distribuiçao normal e sim uma distribuição assimétrica_

`Hb_mean` = 13.41273750879661 <br />
`Hb_median` = 13.2

1. E ao compararmos os valores de hemoglobina dos resultados, encontramos que aqueles que foram possitivos para a Anemia, apresentam valores menores de Hemoglobina quando comparados com aqueles que não apresentaram quadros de anemia

`sns.ecdfplot(data=df, x="Hemoglobin", hue='Result');`

![image](https://user-images.githubusercontent.com/81988636/207468806-89b31bf0-0e88-4298-9677-07f2c8715645.png)

* Nós podemos observar várias plotagens ao mesmo tempo com o comando a seguir

```
g = sns.PairGrid(df)
g.map_diag(sns.histplot, kde=True, color='Green')
g.map_offdiag(sns.scatterplot, color="gray", s=6);
```

![image](https://user-images.githubusercontent.com/81988636/207469467-84c06b57-a7be-4e85-bbda-f99a09da3176.png)

* Também podemos observar o comportamento de todas as variáveis do Eritograma em relação ao `MCHC`

```
fig, ax = plt.subplots(ncols = 3, nrows=1, figsize=(15,5), sharex='col', sharey='row');
plt.subplots_adjust(wspace=0, hspace=0)

# Como as variáveis se relacionam com a Concentração de Hemoglobina
# Encontramos uma relação interessante entre Hemoglobina e o MCHC.
# Vamores baixos de Hb são seguidos de valores baixos de MCHC

sns.lineplot(ax=ax[0], data=df, x='Hemoglobin', y='MCHC', color='red');
sns.lineplot(ax=ax[1], data=df, x='MCH', y='MCHC', color='green');
sns.lineplot(ax=ax[2], data=df, x='MCV', y='MCHC', color='purple');
```

![image](https://user-images.githubusercontent.com/81988636/207469710-12314549-1c4f-4908-8bf7-81c259f6b5e4.png)

Um erro maior entre aqueles que são anêmicos é apresentado por meio deste gráfico

```
sns.lmplot(
    data=df, x="Hemoglobin", y="MCHC",
    fit_reg = True, scatter=True, hue='Result', palette='flare'
);
```

![image](https://user-images.githubusercontent.com/81988636/207470067-828feb32-11c0-4832-9538-835758b7a2e4.png)

### Treinamento do Modelo

`X = df.iloc[:, 0:5].values` <br />
`y = df.iloc[:, 5].values`
