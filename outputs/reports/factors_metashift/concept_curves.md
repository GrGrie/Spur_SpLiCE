# Concept audits of factors_metashift

Detection AUROC: how well the onset score separates attribute units from class units. Types come from hidden labels and are used for this report only.

## Early Concept Swap kill test

**Stage K1 (audit finds the spurious concept): fail.** Mean detection AUROC of `curves` 0.50.

| seed | detection AUROC | type of the leading unit | types of the flagged units | clean flags |
|---|---|---|---|---|
| 1 | 0.50 | untyped | untyped, untyped, attribute | yes |
| 2 | 0.50 | untyped | untyped, untyped, attribute | yes |
| 3 | 0.50 | untyped | untyped, untyped, attribute | yes |
| 4 | 0.50 | untyped | untyped, untyped, attribute | yes |

**Stage K2 (swap raises WGA): fail.** Paired change of validation WGA (last four probes) against `simclr`, in points; a pass needs at least +5 and a gain on three quarters of the seeds, and `curves` above `curves_random` on three quarters of the seeds.

| arm | seeds | mean WGA change | seeds up |
|---|---|---|---|
| curves | 4 | +5.0 | 2/4 |
| curves_bottom | 4 | +3.2 | 2/4 |
| curves_oracle | 4 | +2.9 | 3/4 |
| curves_random | 4 | -1.7 | 1/4 |
| curves_shuffled | 4 | +0.9 | 1/4 |

`curves` against `curves_random`: +6.8 points, up on 4 of 4 seeds.

**Reliance on the flagged units after the swap starts** (summed share; the theory predicts a fall).

| arm | seeds | share at end of early window | share at last audit | seeds where it fell |
|---|---|---|---|---|
| curves | 4 | 0.604 | 0.354 | 4/4 |
| curves_bottom | 4 | 0.113 | 0.062 | 4/4 |
| curves_oracle | 4 | 0.187 | 0.087 | 4/4 |
| curves_random | 4 | 0.568 | 0.385 | 4/4 |
| curves_shuffled | 4 | 0.604 | 0.545 | 4/4 |

## curves

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | top | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; lawn / surfers | surfer | surfing (+1) / pasture | untyped, untyped, attribute |
| 2 | 500 | 0.50 | top | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; lawn / surfers | surfer | surfing (+1) / pasture | untyped, untyped, attribute |
| 3 | 500 | 0.50 | top | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; lawn / surfers | surfer | surfing (+1) / pasture | untyped, untyped, attribute |
| 4 | 500 | 0.50 | top | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; lawn / surfers | surfer | surfing (+1) / pasture | untyped, untyped, attribute |

| unit | words | presence | area | mean onset | hidden-label type |
|---|---|---|---|---|---|
| 0 | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | 1.00 | 0.61 | 0.347 | untyped |
| 2 | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | 0.12 | 0.24 | 0.134 | untyped |
| 3 | lawn / surfers | surfer | surfing (+1) / pasture | 0.31 | 0.22 | 0.133 | attribute |
| 10 | chats | chat | cats / kitty | kitten | cat / kittens | 0.48 | 0.19 | 0.100 | class |
| 7 | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | 0.20 | 0.21 | 0.093 | untyped |
| 11 | beaches | 0.07 | 0.22 | 0.087 | untyped |
| 4 | pups | pup | puppies (+2) / goats | goat / sheep | 0.52 | 0.11 | 0.049 | class |
| 5 | laptops | laptop / desks | desk / beds | bed | 0.42 | 0.15 | 0.046 | attribute |
| 1 | canine | dog / husky / retriever | 0.33 | 0.05 | 0.011 | untyped |

## curves_bottom

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | bottom | canine | dog / husky / retriever; pups | pup | puppies (+2) / goats | goat / sheep; laptops | laptop / desks | desk / beds | bed | untyped, class, attribute |
| 2 | 500 | 0.50 | bottom | canine | dog / husky / retriever; pups | pup | puppies (+2) / goats | goat / sheep; laptops | laptop / desks | desk / beds | bed | untyped, class, attribute |
| 3 | 500 | 0.50 | bottom | canine | dog / husky / retriever; pups | pup | puppies (+2) / goats | goat / sheep; laptops | laptop / desks | desk / beds | bed | untyped, class, attribute |
| 4 | 500 | 0.50 | bottom | canine | dog / husky / retriever; pups | pup | puppies (+2) / goats | goat / sheep; laptops | laptop / desks | desk / beds | bed | untyped, class, attribute |

| unit | words | presence | area | mean onset | hidden-label type |
|---|---|---|---|---|---|
| 0 | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | 1.00 | 0.61 | 0.347 | untyped |
| 2 | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | 0.12 | 0.24 | 0.134 | untyped |
| 3 | lawn / surfers | surfer | surfing (+1) / pasture | 0.31 | 0.22 | 0.133 | attribute |
| 10 | chats | chat | cats / kitty | kitten | cat / kittens | 0.48 | 0.19 | 0.100 | class |
| 7 | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | 0.20 | 0.21 | 0.093 | untyped |
| 11 | beaches | 0.07 | 0.22 | 0.087 | untyped |
| 4 | pups | pup | puppies (+2) / goats | goat / sheep | 0.52 | 0.11 | 0.049 | class |
| 5 | laptops | laptop / desks | desk / beds | bed | 0.42 | 0.15 | 0.046 | attribute |
| 1 | canine | dog / husky / retriever | 0.33 | 0.05 | 0.011 | untyped |

## curves_oracle

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | oracle | lawn / surfers | surfer | surfing (+1) / pasture; laptops | laptop / desks | desk / beds | bed | attribute, attribute |
| 2 | 500 | 0.50 | oracle | lawn / surfers | surfer | surfing (+1) / pasture; laptops | laptop / desks | desk / beds | bed | attribute, attribute |
| 3 | 500 | 0.50 | oracle | lawn / surfers | surfer | surfing (+1) / pasture; laptops | laptop / desks | desk / beds | bed | attribute, attribute |
| 4 | 500 | 0.50 | oracle | lawn / surfers | surfer | surfing (+1) / pasture; laptops | laptop / desks | desk / beds | bed | attribute, attribute |

| unit | words | presence | area | mean onset | hidden-label type |
|---|---|---|---|---|---|
| 0 | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | 1.00 | 0.61 | 0.347 | untyped |
| 2 | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | 0.12 | 0.24 | 0.134 | untyped |
| 3 | lawn / surfers | surfer | surfing (+1) / pasture | 0.31 | 0.22 | 0.133 | attribute |
| 10 | chats | chat | cats / kitty | kitten | cat / kittens | 0.48 | 0.19 | 0.100 | class |
| 7 | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | 0.20 | 0.21 | 0.093 | untyped |
| 11 | beaches | 0.07 | 0.22 | 0.087 | untyped |
| 4 | pups | pup | puppies (+2) / goats | goat / sheep | 0.52 | 0.11 | 0.049 | class |
| 5 | laptops | laptop / desks | desk / beds | bed | 0.42 | 0.15 | 0.046 | attribute |
| 1 | canine | dog / husky / retriever | 0.33 | 0.05 | 0.011 | untyped |

## curves_random

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | random | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; beaches | untyped, untyped, untyped |
| 2 | 500 | 0.50 | random | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; beaches | untyped, untyped, untyped |
| 3 | 500 | 0.50 | random | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; beaches | untyped, untyped, untyped |
| 4 | 500 | 0.50 | random | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; beaches | untyped, untyped, untyped |

| unit | words | presence | area | mean onset | hidden-label type |
|---|---|---|---|---|---|
| 0 | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | 1.00 | 0.61 | 0.347 | untyped |
| 2 | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | 0.12 | 0.24 | 0.134 | untyped |
| 3 | lawn / surfers | surfer | surfing (+1) / pasture | 0.31 | 0.22 | 0.133 | attribute |
| 10 | chats | chat | cats / kitty | kitten | cat / kittens | 0.48 | 0.19 | 0.100 | class |
| 7 | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | 0.20 | 0.21 | 0.093 | untyped |
| 11 | beaches | 0.07 | 0.22 | 0.087 | untyped |
| 4 | pups | pup | puppies (+2) / goats | goat / sheep | 0.52 | 0.11 | 0.049 | class |
| 5 | laptops | laptop / desks | desk / beds | bed | 0.42 | 0.15 | 0.046 | attribute |
| 1 | canine | dog / husky / retriever | 0.33 | 0.05 | 0.011 | untyped |

## curves_shuffled

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | top | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; lawn / surfers | surfer | surfing (+1) / pasture | untyped, untyped, attribute |
| 2 | 500 | 0.50 | top | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; lawn / surfers | surfer | surfing (+1) / pasture | untyped, untyped, attribute |
| 3 | 500 | 0.50 | top | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; lawn / surfers | surfer | surfing (+1) / pasture | untyped, untyped, attribute |
| 4 | 500 | 0.50 | top | blankets | blanket / seaside | coastal | ocean (+1) / farm | field; swim | swimming / boating | boats | boat / fishermen | fisherman | fishing; lawn / surfers | surfer | surfing (+1) / pasture | untyped, untyped, attribute |

| unit | words | presence | area | mean onset | hidden-label type |
|---|---|---|---|---|---|
| 0 | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | 1.00 | 0.61 | 0.347 | untyped |
| 2 | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | 0.12 | 0.24 | 0.134 | untyped |
| 3 | lawn / surfers | surfer | surfing (+1) / pasture | 0.31 | 0.22 | 0.133 | attribute |
| 10 | chats | chat | cats / kitty | kitten | cat / kittens | 0.48 | 0.19 | 0.100 | class |
| 7 | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | 0.20 | 0.21 | 0.093 | untyped |
| 11 | beaches | 0.07 | 0.22 | 0.087 | untyped |
| 4 | pups | pup | puppies (+2) / goats | goat / sheep | 0.52 | 0.11 | 0.049 | class |
| 5 | laptops | laptop / desks | desk / beds | bed | 0.42 | 0.15 | 0.046 | attribute |
| 1 | canine | dog / husky / retriever | 0.33 | 0.05 | 0.011 | untyped |
