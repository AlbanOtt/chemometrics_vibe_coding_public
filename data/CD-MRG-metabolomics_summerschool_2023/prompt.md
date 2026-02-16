## Project: Non-invasive drug monitoring through skin metabolomics

Non-invasive drug monitoring is desirable as it improves patient experience. Instead of relying on invasive blood draws drug pharmacokinetics and metabolism could for example be monitored through the skin. We will be working with a subset of the data published by [Panitchpakdi and collaborators (2022)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0271794). The dataset comes from [CD-MRG-metabolomics_summerschool_2023](https://github.com/ssi-dk/CD-MRG-metabolomics_summerschool_2023), it consists of plasma and skin swabs (forearm, forehead) collected from healthy volunteers (n=7) over the course of 24 hours, who have been administered the antihistaminic diphenhydramine. Each skin area (forearm, forehead) must be analyzed separately. Ignore the upper back skin swabs for this project.
The measurements for forearm and forehead skin are in separated filed, both contains plasma samples.

Your task will be to investigate whether:

- Diphenhydramine and its metabolites can be detected in the skin and whether it exhibits similar pharmacokinetics as in plasma?
- You can observe other metabolites that exhibit interesting time trends in plasma and whether those metabolites can also be detected in the skin?
- Identify one or more skin metabolite(s) that can be used as a proxy to monitor Diphenhydramine levels in plasma.
- What is the best skin area to monitor Diphenhydramine levels in plasma?

You take it from the begining, and you are expected to perform all the necessary data processing and analysis steps to answer the questions above. Use python and the libraries that are most appropriate for metabolomics data analysis. You are expected to write a report describing your analysis and results in quarto, and to present your findings in a clear and concise manner.
Write everything you understand about the project and the data in the report, and make sure to include all the necessary details for someone else to reproduce your analysis.

files are here:
data\CD-MRG-metabolomics_summerschool_2023