> E.g., "the user's interest can be fully captured merely with those representative keywords", therefore, further experiments are needed to prove these points or discuss the trade-off between effectiveness and efficiency
- experiments of effectiveness-efficiency trade-off

> a number of important baseline models and related work are missing. 1) considering the solutions reported on htps://msnews.github.io. 2) Comparing with the pre-computed and cached solutions, non-personalized keyword extraction methods
- donot claim we achieved state-of-the-art performance
- non-personalized selection

> some of the descriptions are unclear e.g. how to process users with long browsing history;
> For the reader it seems e.g. somewhat unsatisfactory that we need two different user profiles The keywords extracte from each news item are personalized and later a user embedding is computed form these personalized keywords.
- two tower model can process super long user browsing history

> In my opinion, the authors should design a new method/framework to speed up news feed recommendation instead of just decreasing the input information.
- our new motivation is: content-based recommendation often faces the challenge to truncate input, which may lead to severe effectiveness loss, we propose a new model to intellegently filter input

> The improvement in effectiveness and efficiency is not very strong
- we do not need to emphesize efficiency improvement, instead we just claim our model is better than truncation and heuristics

- overall performance
  - v.s. baselines
- **k v.s. AUC**
- variant models
- sparse transformers
- different input length and different compression ratio

> The description of teh variants of the GF model described 428-437 is very unclear and should be describe with some more detail. E.g.: "where the user-side in-put is filtered by BM25 score" at which point the filtering is done
- make the statement clear

### Baselines
1. **Truncation**
   - First-k token
2. **Entity**
   - First-k Entity
   - BM25-k Entity
3. **Heuristic**
   - BM25-k token
4. **Dynamic**
   - BM25H-k token
5. **Unsupervised Keyword Extraction**
   - KeyBERT-k
6. **Candidate-Aware?**> E.g., "the user's interest can be fully captured merely with those representative keywords", therefore, further experiments are needed to prove these points or discuss the trade-off between effectiveness and efficiency
- experiments of effectiveness-efficiency trade-off

> a number of important baseline models and related work are missing. 1) considering the solutions reported on htps://msnews.github.io. 2) Comparing with the pre-computed and cached solutions, non-personalized keyword extraction methods
- donot claim we achieved state-of-the-art performance
- non-personalized selection

> some of the descriptions are unclear e.g. how to process users with long browsing history;
> For the reader it seems e.g. somewhat unsatisfactory that we need two different user profiles The keywords extracte from each news item are personalized and later a user embedding is computed form these personalized keywords.
- two tower model can process super long user browsing history

> In my opinion, the authors should design a new method/framework to speed up news feed recommendation instead of just decreasing the input information.
- our new motivation is: content-based recommendation often faces the challenge to truncate input, which may lead to severe effectiveness loss, we propose a new model to intellegently filter input

> The improvement in effectiveness and efficiency is not very strong
- we do not need to emphesize efficiency improvement, instead we just claim our model is better than truncation and heuristics

- overall performance
  - v.s. baselines
- **k v.s. AUC**
- variant models
- sparse transformers
- different input length and different compression ratio

> The description of teh variants of the GF model described 428-437 is very unclear and should be describe with some more detail. E.g.: "where the user-side in-put is filtered by BM25 score" at which point the filtering is done
- make the statement clear

### Baselines
1. **Truncation**
   - First-k token
2. **Entity**
   - First-k Entity
   - BM25-k Entity
3. **Heuristic**
   - BM25-k token
4. **Dynamic**
   - BM25H-k token
5. **Unsupervised Keyword Extraction**
   - KeyBERT-k
6. **Candidate-Aware?**