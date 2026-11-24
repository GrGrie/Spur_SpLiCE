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

## Concept attribution

**Outcome: weak.** H1 (an attribute unit is a shortcut and a class unit is needed) fails; H2 (Spearman of onset and effect at least 0.5) holds; H3 (the swapped unit's own share falls) holds, on 9 of 9 units. Fewest seeds per unit: 4; the fail outcome needs every unit neutral at 8 seeds.

Each row swaps one concept unit alone after the early window (arm `curves_unit_<k>`) and reports the paired change of validation WGA against `simclr`, in points. Spearman correlation of onset score and WGA effect across units: 0.60. Verdict: shortcut or needed when the sign holds on three quarters of the seeds and the mean is at least twice its standard error, neutral otherwise. The column *vs unit arms* compares each unit arm with the mean of all unit arms of the same seed.

Does the swapped unit matter? Unit-by-seed F test over 4 seeds: F = 0.76, p = 0.64, run-to-run sd 4.1 points.

| onset rank | unit | words | onset | area | hidden-label type | WGA effect | sd | seeds up | verdict | vs unit arms | accuracy effect | own share, epoch 50 to last, and its reference (seeds below it) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | 0.347 | 0.61 | untyped | +5.2 | 12.4 | 2/4 | neutral | +1.8 (2/4 up) | +6.9 | 0.330 to 0.188, reference 0.330 (4/4) |
| 2 | 2 | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | 0.134 | 0.24 | untyped | +3.7 | 8.1 | 3/4 | neutral | +0.3 (2/4 up) | +2.6 | 0.137 to 0.089, reference 0.137 (4/4) |
| 3 | 3 | lawn / surfers | surfer | surfing (+1) / pasture | 0.133 | 0.22 | attribute | +5.2 | 9.4 | 2/4 | neutral | +1.8 (4/4 up) | +2.3 | 0.137 to 0.055, reference 0.137 (4/4) |
| 4 | 10 | chats | chat | cats / kitty | kitten | cat / kittens | 0.100 | 0.19 | class | +4.5 | 7.9 | 2/4 | neutral | +1.1 (2/4 up) | +2.5 | 0.094 to 0.034, reference 0.094 (4/4) |
| 5 | 7 | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | 0.093 | 0.21 | untyped | +2.7 | 10.9 | 2/4 | neutral | -0.8 (2/4 up) | +1.3 | 0.088 to 0.054, reference 0.088 (4/4) |
| 6 | 11 | beaches | 0.087 | 0.22 | untyped | -0.6 | 10.9 | 1/4 | neutral | -4.1 (1/4 up) | -1.0 | 0.100 to 0.084, reference 0.100 (4/4) |
| 7 | 4 | pups | pup | puppies (+2) / goats | goat / sheep | 0.049 | 0.11 | class | +3.8 | 8.8 | 2/4 | neutral | +0.4 (1/4 up) | +1.0 | 0.051 to 0.023, reference 0.051 (4/4) |
| 8 | 5 | laptops | laptop / desks | desk / beds | bed | 0.046 | 0.15 | attribute | +2.9 | 7.7 | 2/4 | neutral | -0.6 (1/4 up) | +1.2 | 0.050 to 0.036, reference 0.050 (4/4) |
| 9 | 1 | canine | dog / husky / retriever | 0.011 | 0.05 | untyped | +3.6 | 7.8 | 3/4 | neutral | +0.1 (2/4 up) | +1.7 | 0.012 to 0.011, reference 0.012 (3/4) |

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

## curves_unit_0

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | units | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | untyped |
| 2 | 500 | 0.50 | units | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | untyped |
| 3 | 500 | 0.50 | units | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | untyped |
| 4 | 500 | 0.50 | units | blankets | blanket / seaside | coastal | ocean (+1) / farm | field | untyped |

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

## curves_unit_1

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | units | canine | dog / husky / retriever | untyped |
| 2 | 500 | 0.50 | units | canine | dog / husky / retriever | untyped |
| 3 | 500 | 0.50 | units | canine | dog / husky / retriever | untyped |
| 4 | 500 | 0.50 | units | canine | dog / husky / retriever | untyped |

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

## curves_unit_10

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | units | chats | chat | cats / kitty | kitten | cat / kittens | class |
| 2 | 500 | 0.50 | units | chats | chat | cats / kitty | kitten | cat / kittens | class |
| 3 | 500 | 0.50 | units | chats | chat | cats / kitty | kitten | cat / kittens | class |
| 4 | 500 | 0.50 | units | chats | chat | cats / kitty | kitten | cat / kittens | class |

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

## curves_unit_11

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | units | beaches | untyped |
| 2 | 500 | 0.50 | units | beaches | untyped |
| 3 | 500 | 0.50 | units | beaches | untyped |
| 4 | 500 | 0.50 | units | beaches | untyped |

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

## curves_unit_2

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | units | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | untyped |
| 2 | 500 | 0.50 | units | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | untyped |
| 3 | 500 | 0.50 | units | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | untyped |
| 4 | 500 | 0.50 | units | swim | swimming / boating | boats | boat / fishermen | fisherman | fishing | untyped |

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

## curves_unit_3

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | units | lawn / surfers | surfer | surfing (+1) / pasture | attribute |
| 2 | 500 | 0.50 | units | lawn / surfers | surfer | surfing (+1) / pasture | attribute |
| 3 | 500 | 0.50 | units | lawn / surfers | surfer | surfing (+1) / pasture | attribute |
| 4 | 500 | 0.50 | units | lawn / surfers | surfer | surfing (+1) / pasture | attribute |

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

## curves_unit_4

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | units | pups | pup | puppies (+2) / goats | goat / sheep | class |
| 2 | 500 | 0.50 | units | pups | pup | puppies (+2) / goats | goat / sheep | class |
| 3 | 500 | 0.50 | units | pups | pup | puppies (+2) / goats | goat / sheep | class |
| 4 | 500 | 0.50 | units | pups | pup | puppies (+2) / goats | goat / sheep | class |

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

## curves_unit_5

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | units | laptops | laptop / desks | desk / beds | bed | attribute |
| 2 | 500 | 0.50 | units | laptops | laptop / desks | desk / beds | bed | attribute |
| 3 | 500 | 0.50 | units | laptops | laptop / desks | desk / beds | bed | attribute |
| 4 | 500 | 0.50 | units | laptops | laptop / desks | desk / beds | bed | attribute |

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

## curves_unit_7

Mean detection AUROC 0.50, seed agreement of the onset scores (mean Spearman) 0.99.

| seed | audited to epoch | detection AUROC | rule | flagged units | types |
|---|---|---|---|---|---|
| 1 | 500 | 0.50 | units | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | untyped |
| 2 | 500 | 0.50 | units | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | untyped |
| 3 | 500 | 0.50 | units | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | untyped |
| 4 | 500 | 0.50 | units | drove | drivers | driving (+1) / bicycles | biking | biker (+5) / motorbike | motorcycles | motorcycle | untyped |

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
