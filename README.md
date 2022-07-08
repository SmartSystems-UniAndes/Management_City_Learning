# City Learning

## About

This repository contains a study case of the work developed by Kathirgamanathan, A. et al. in * A Centralised Soft Actor
Critic Deep Reinforcement Learning Approach to District Demand Side Management through CityLearn* [1] and by 
Vazquez-Canteli, J. in CityLearn: Standardizing Research in Multi-Agent Reinforcement Learning for Demand Response and
Urban Energy Management [2]. Also, is based on the GitHub public repository *Actor-Critic-Methods-Paper-To-Code* [3].

## Work Environment

To use this repository it is essential to clone the public repository 
[*Actor-Critic-Methods-Paper-To-Code*](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code). 

## How it works?

Add the [*reward_function.py*](reward_function.py) and 
[*central-agent-ddpg-tensorflow.ipynb*](central-agent-ddpg-tensorflow.ipynb) file, and also the
[*DDPG_Agent*](DDPG_Agent) folder to the cloned repository. Once done that, run the 
[*central-agent-ddpg-tensorflow.ipynb*](central-agent-ddpg-tensorflow.ipynb) file, which is a self content file to see
the trainin process of the DDPG agent.

## Citing Work

```BibTeX

@article{gaviria_machine_2022,
	title = {Machine learning in photovoltaic systems: A review},
	issn = {0960-1481},
	url = {https://www.sciencedirect.com/science/article/pii/S0960148122009454},
	doi = {10.1016/j.renene.2022.06.105},
	shorttitle = {Machine learning in photovoltaic systems},
	abstract = {This paper presents a review of up-to-date Machine Learning ({ML}) techniques applied to photovoltaic ({PV}) systems, with a special focus on deep learning. It examines the use of {ML} applied to control, islanding detection, management, fault detection and diagnosis, forecasting irradiance and power generation, sizing, and site adaptation in {PV} systems. The contribution of this work is three fold: first, we review more than 100 research articles, most of them from the last five years, that applied state-of-the-art {ML} techniques in {PV} systems; second, we review resources where researchers can find open data-sets, source code, and simulation environments that can be used to test {ML} algorithms; third, we provide a case study for each of one of the topics with open-source code and data to facilitate researchers interested in learning about these topics to introduce themselves to implementations of up-to-date {ML} techniques applied to {PV} systems. Also, we provide some directions, insights, and possibilities for future development.},
	journaltitle = {Renewable Energy},
	shortjournal = {Renewable Energy},
	author = {Gaviria, Jorge Felipe and Narváez, Gabriel and Guillen, Camilo and Giraldo, Luis Felipe and Bressan, Michael},
	urldate = {2022-07-03},
	date = {2022-07-01},
	langid = {english},
	keywords = {Deep learning, Machine learning, Neural networks, Photovoltaic systems, Reinforcement learning, Review},
	file = {ScienceDirect Snapshot:C\:\\Users\\jfgf1\\Zotero\\storage\\G96H46L2\\S0960148122009454.html:text/html},
},


@inproceedings{kathirgamanathan2020centralised,
  title={A Centralised Soft Actor Critic Deep Reinforcement Learning Approach to District Demand Side Management through CityLearn},
  author={Kathirgamanathan, Anjukan and Twardowski, Kacper and Mangina, Eleni and Finn, Donal P},
  booktitle={Proceedings of the 1st International Workshop on Reinforcement Learning for Energy Management in Buildings \& Cities},
  pages={11--14},
  year={2020}
}

@inproceedings{kathirgamanathan2020centralised,
  title={A Centralised Soft Actor Critic Deep Reinforcement Learning Approach to District Demand Side Management through CityLearn},
  author={Kathirgamanathan, Anjukan and Twardowski, Kacper and Mangina, Eleni and Finn, Donal P},
  booktitle={Proceedings of the 1st International Workshop on Reinforcement Learning for Energy Management in Buildings \& Cities},
  pages={11--14},
  year={2020}
}
```

## References
[1] Jorge Felipe Gaviria, Gabriel Narváez, Camilo Guillen, Luis Felipe Giraldo, and Michael Bressan. Machine learning in photovoltaic systems: A review. ISSN 0960-1481. doi: 10.1016/j.renene.2022.06.105. URL https://www.sciencedirect.com/science/article/pii/S0960148122009454?via%3Dihub
[2] Kathirgamanathan, A., Twardowski, K., Mangina, E., & Finn, D. P. (2020, November). *A Centralised Soft Actor Critic 
Deep Reinforcement Learning Approach to District Demand Side Management through CityLearn*. In Proceedings of the 1st 
International Workshop on Reinforcement Learning for Energy Management in Buildings & Cities (pp. 11-14).
[3] Vazquez-Canteli, J. R., Dey, S., Henze, G., & Nagy, Z. (2020). *CityLearn: Standardizing Research in Multi-Agent
Reinforcement Learning for Demand Response and Urban Energy Management*. arXiv preprint arXiv:2012.10504.
[4] Tabor, P. (2020). *Actor-Critic-Methods-Paper-To-Code* [Source code]. 
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code.

## Licenses

### Software
The software is licensed under an [MIT License](https://opensource.org/licenses/MIT). A copy of the license has been included in the repository and can be found [here](https://github.com/SmartSystems-UniAndes/PV_MPPT_Control_Based_on_Reinforcement_Learning/blob/main/LICENSE-MIT.txt).

