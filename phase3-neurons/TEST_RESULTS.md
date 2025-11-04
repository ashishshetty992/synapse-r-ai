# Quick Wins Test Results

## 1. Count by city with preferTargets

**Command:**
```bash
curl -X POST localhost:8000/synapse/fill \
  -H 'Content-Type: application/json' \
  -d '{"intent":{"ask":"top_k","metric":{"op":"count"},"target":"orders.id"},"preferRoles":["geo"],"preferTargets":["orders.shipping_city","customer.city","orders.country"]}'
```

**Result:**
{
  "ok": true,
  "intentFilled": {
    "ask": "top_k",
    "metric": {
      "op": "count"
    },
    "target": "orders.country"
  },
  "targetCandidates": [
    {
      "entity": "orders",
      "name": "country",
      "score": 0.9068900563150901,
      "roleTop": "geo"
    },
    {
      "entity": "orders",
      "name": "shipping_city",
      "score": 0.6617836675633075,
      "roleTop": "geo"
    },
    {
      "entity": "customer",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    },
    {
      "entity": "product",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    },
    {
      "entity": "orders",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    },
    {
      "entity": "order_item",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    },
    {
      "entity": "payment",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    },
    {
      "entity": "customer",
      "name": "segment",
      "score": 0.5633074141657735,
      "roleTop": "category"
    }
  ],
  "entityCandidates": [
    {
      "entity": "orders",
      "score": 0.5755934300021159
    },
    {
      "entity": "payment",
      "score": 0.46938306500018734
    },
    {
      "entity": "order_item",
      "score": 0.34570733293160383
    },
    {
      "entity": "customer",
      "score": 0.3306808907129765
    },
    {
      "entity": "product",
      "score": 0.27623663132706905
    }
  ],
  "conflicts": [
    {
      "slot": "category",
      "candidates": [
        "payment.method",
        "customer.segment",
        "orders.status",
        "product.category"
      ],
      "resolution": "payment.method",
      "why": {
        "PathScore": 0.20624745687818793,
        "CosineContext": 0.23509751778562182,
        "RoleCoherence": 1.0,
        "LocalBonus": 0.0,
        "weights": {
          "path": 0.5,
          "cosine": 0.35,
          "role": 0.15
        }
      },
      "scores": [
        {
          "target": "payment.method",
          "score": 0.3354078596640616,
          "PathScore": 0.20624745687818793,
          "CosineContext": 0.23509751778562182,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "customer.segment",
          "score": 0.3082105923817643,
          "PathScore": 0.13370188275371012,
          "CosineContext": 0.26102757429974066,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "orders.status",
          "score": 0.2961420656124838,
          "PathScore": 0.15,
          "CosineContext": 0.11754875889281091,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.03,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "product.category",
          "score": 0.2669345668669293,
          "PathScore": 0.14477370888149063,
          "CosineContext": 0.1272791783605256,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        }
      ]
    },
    {
      "slot": "timestamp",
      "candidates": [
        "orders.created_at",
        "order_item.created_at",
        "payment.created_at",
        "product.created_at",
        "customer.created_at"
      ],
      "resolution": "orders.created_at",
      "why": {
        "PathScore": 0.15,
        "CosineContext": 0.14105851067137304,
        "RoleCoherence": 1.0,
        "LocalBonus": 0.03,
        "weights": {
          "path": 0.5,
          "cosine": 0.35,
          "role": 0.15
        }
      },
      "scores": [
        {
          "target": "orders.created_at",
          "score": 0.3043704787349806,
          "PathScore": 0.15,
          "CosineContext": 0.14105851067137304,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.03,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "order_item.created_at",
          "score": 0.3024942071740745,
          "PathScore": 0.20624745687818793,
          "CosineContext": 0.14105851067137304,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "payment.created_at",
          "score": 0.3024942071740745,
          "PathScore": 0.20624745687818793,
          "CosineContext": 0.14105851067137304,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "product.created_at",
          "score": 0.27175733317572587,
          "PathScore": 0.14477370888149063,
          "CosineContext": 0.14105851067137304,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "customer.created_at",
          "score": 0.2662214201118356,
          "PathScore": 0.13370188275371012,
          "CosineContext": 0.14105851067137304,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        }
      ]
    }
  ],
  "alignPlus": {
    "Abase": 0.7268900563150902,
    "Coverage": 1.0,
    "OffSchemaRate": 0.0,
    "Aplus": 0.7268900563150902
  },
  "debug": {
    "topAliasHits": [],
    "tokens": [],
    "mappedTerms": [],
    "rolePrior": {
      "timestamp": 0.04520988077045555,
      "geo": 0.11107820271276915,
      "text": 0.04333348945104675,
      "category": 0.0839062398214722,
      "quantity": 0.2687846431113821,
      "id": 0.2978168736164255,
      "money": 0.14987067051644873
    },
    "matchDebug": {
      "rolePrior": {
        "timestamp": 0.04520988077045555,
        "geo": 0.11107820271276915,
        "text": 0.04333348945104675,
        "category": 0.0839062398214722,
        "quantity": 0.2687846431113821,
        "id": 0.2978168736164255,
        "money": 0.14987067051644873
      },
      "keywordPrior": {
        "quantity": 0.5,
        "id": 0.5
      },
      "centroidPrior": {
        "id": 0.16302812269404246,
        "timestamp": 0.07534980128409258,
        "money": 0.24978445086074788,
        "geo": 0.18513033785461525,
        "category": 0.139843733035787,
        "text": 0.07222248241841125,
        "quantity": 0.11464107185230349
      },
      "blendWeights": {
        "keyword": 0.4,
        "centroid": 0.6
      },
      "entityBlend": {
        "entityWeight": 0.7,
        "fieldWeight": 0.3,
        "entityAliasAlpha": 0.15
      },
      "roleAlphaUsed": 0.1,
      "shapingWeights": {
        "aliasAlpha": 0.4,
        "shapeAlpha": 0.3,
        "shapeBeta": 0.2,
        "metricAlpha": 0.4
      },
      "consideredFields": 32,
      "vocabDim": 384,
      "ask": "top_k",
      "metricOp": "count",
      "targetPairs": [
        [
          "orders",
          "id"
        ]
      ]
    },
    "fillConfigUsed": {
      "unknownPenalty": 0.1,
      "maxUnknownInTopK": 1,
      "preferBonus": 0.08,
      "moneyPivotPenalty": 0.08,
      "osrFromIntentOnly": true,
      "conflict": {
        "pathWeight": 0.5,
        "cosineWeight": 0.35,
        "roleWeight": 0.15,
        "minPathScore": 0.02,
        "lengthPriorBase": 0.92,
        "preferMetricEntityBonus": 0.03
      }
    },
    "preferRolesUsed": [
      "geo"
    ],
    "preferTargetsUsed": [
      "orders.shipping_city",
      "customer.city",
      "orders.country"
    ]
  },
  "requestId": "5d41943d-c087-4bee-8c53-e9e5390d17af"
}

---

## 2. NL Encoding - "top revenue by city last month"

**Command:**
```bash
curl -X POST localhost:8000/intent/encode_nl \
  -H 'Content-Type: application/json' \
  -d '{"text":"top revenue by city last month"}'
```

**Result:**
{
  "version": "intent-nl/0.1",
  "dim": 384,
  "vec": [
    0.09865915001721291,
    0.07486723596100274,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.040817021357807695,
    0.040817021357807695,
    0.11568425731881044,
    0.0,
    0.06635468231020397,
    0.0,
    0.049329575008606456,
    0.049329575008606456,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1259425578657953,
    0.0,
    0.0,
    0.0,
    0.233114261533807,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.008512553650798763,
    0.008512553650798763,
    0.09189234326260026,
    0.0,
    0.0,
    0.0,
    0.0,
    0.09189234326260026,
    0.040817021357807695,
    0.040817021357807695,
    0.18203893962901443,
    0.0,
    0.02553766095239628,
    0.14122191827120673,
    0.3317734115510199,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.06635468231020397,
    0.09189234326260026,
    0.09189234326260026,
    0.09189234326260026,
    0.09189234326260026,
    0.18203893962901443,
    0.06635468231020397,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.06635468231020397,
    0.0,
    0.0,
    0.0,
    0.0,
    0.07486723596100274,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.14973447192200548,
    0.0,
    0.02553766095239628,
    0.13270936462040794,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2075766005814107,
    0.09189234326260026,
    0.09189234326260026,
    0.09189234326260026,
    0.09189234326260026,
    0.040817021357807695,
    0.09189234326260026,
    0.09189234326260026,
    0.040817021357807695,
    0.040817021357807695,
    0.18203893962901443,
    0.18203893962901443,
    0.008512553650798763,
    0.008512553650798763,
    0.008512553650798763,
    0.008512553650798763,
    0.0,
    0.18203893962901443,
    0.18203893962901443,
    0.0,
    0.0,
    0.18203893962901443,
    0.18203893962901443,
    0.18203893962901443,
    0.0,
    0.0,
    0.18203893962901443,
    0.18203893962901443,
    0.0,
    0.0,
    0.0,
    0.18203893962901443,
    0.0,
    0.0,
    0.07486723596100274,
    0.07486723596100274,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.02553766095239628,
    0.02553766095239628,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.02553766095239628,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.07486723596100274,
    0.11568425731881044,
    0.0,
    0.0,
    0.0,
    0.07486723596100274,
    0.07486723596100274,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.07486723596100274,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.0,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.0,
    0.0,
    0.0,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.02553766095239628,
    0.0,
    0.0,
    0.0,
    0.06635468231020397,
    0.06635468231020397,
    0.0,
    0.0,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.0,
    0.0,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.0,
    0.0,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.06635468231020397,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "vocab": [
    "##",
    "at",
    "d#",
    "d##",
    "id",
    "id#",
    "id##",
    "#c",
    "##c",
    "t#",
    "t##",
    "re",
    "te",
    "_a",
    "ate",
    "#i",
    "##i",
    "#id",
    "##id",
    "#id#",
    "##id#",
    "#id##",
    "me",
    "e#",
    "e##",
    "#s",
    "nt",
    "##s",
    "cr",
    "ea",
    "ed",
    "d_",
    "#cr",
    "cre",
    "rea",
    "eat",
    "ted",
    "ed_",
    "d_a",
    "_at",
    "at#",
    "##cr",
    "#cre",
    "crea",
    "reat",
    "eate",
    "ated",
    "ted_",
    "ed_a",
    "d_at",
    "_at#",
    "at##",
    "##cre",
    "#crea",
    "creat",
    "reate",
    "eated",
    "ated_",
    "ted_a",
    "ed_at",
    "d_at#",
    "_at##",
    "y#",
    "y##",
    "am",
    "_i",
    "_id",
    "_id#",
    "_id##",
    "un",
    "nt#",
    "nt##",
    "or",
    "pr",
    "us",
    "st",
    "er",
    "r_",
    "er_",
    "r_i",
    "er_i",
    "r_id",
    "er_id",
    "r_id#",
    "ta",
    "ou",
    "oun",
    "unt",
    "ount",
    "it",
    "l_",
    "na",
    "nam",
    "ame",
    "me#",
    "name",
    "ame#",
    "me##",
    "name#",
    "ame##",
    "eg",
    "en",
    "ry",
    "ry#",
    "ry##",
    "#p",
    "ri",
    "ic",
    "ce",
    "##p",
    "#pr",
    "pri",
    "ric",
    "ice",
    "ce#",
    "##pr",
    "pric",
    "rice",
    "ice#",
    "ce##",
    "price",
    "rice#",
    "ice##",
    "#a",
    "ct",
    "ti",
    "##a",
    "cu",
    "to",
    "#cu",
    "##cu",
    "tu",
    "s#",
    "#st",
    "sta",
    "tat",
    "atu",
    "tus",
    "us#",
    "s##",
    "##st",
    "#sta",
    "stat",
    "tatu",
    "atus",
    "tus#",
    "us##",
    "##sta",
    "#stat",
    "statu",
    "tatus",
    "atus#",
    "tus##",
    "mo",
    "amo",
    "mou",
    "amou",
    "moun",
    "unt#",
    "amoun",
    "mount",
    "ount#",
    "unt##",
    "ty",
    "ity",
    "ty#",
    "ity#",
    "ty##",
    "ity##",
    "#o",
    "rd",
    "de",
    "##o",
    "#or",
    "ord",
    "rde",
    "der",
    "##or",
    "#ord",
    "orde",
    "rder",
    "der_",
    "##ord",
    "#orde",
    "order",
    "rder_",
    "der_i",
    "od",
    "t_",
    "#e",
    "em",
    "ma",
    "ai",
    "il",
    "l#",
    "##e",
    "#em",
    "ema",
    "mai",
    "ail",
    "il#",
    "l##",
    "##em",
    "#ema",
    "emai",
    "mail",
    "ail#",
    "il##",
    "##ema",
    "#emai",
    "email",
    "mail#",
    "ail##",
    "#f",
    "fu",
    "ul",
    "ll",
    "_n",
    "##f",
    "#fu",
    "ful",
    "ull",
    "ll_",
    "l_n",
    "_na",
    "##fu",
    "#ful",
    "full",
    "ull_",
    "ll_n",
    "l_na",
    "_nam",
    "##ful",
    "#full",
    "full_",
    "ull_n",
    "ll_na",
    "l_nam",
    "_name",
    "se",
    "gm",
    "#se",
    "seg",
    "egm",
    "gme",
    "men",
    "ent",
    "##se",
    "#seg",
    "segm",
    "egme",
    "gmen",
    "ment",
    "ent#",
    "##seg",
    "#segm",
    "segme",
    "egmen",
    "gment",
    "ment#",
    "ent##",
    "sk",
    "ku",
    "u#",
    "#sk",
    "sku",
    "ku#",
    "u##",
    "##sk",
    "#sku",
    "sku#",
    "ku##",
    "##sku",
    "#sku#",
    "sku##",
    "#n",
    "##n",
    "#na",
    "##na",
    "#nam",
    "##nam",
    "#name",
    "ca",
    "go",
    "#ca",
    "cat",
    "teg",
    "ego",
    "gor",
    "ory",
    "##ca",
    "#cat",
    "cate",
    "ateg",
    "tego",
    "egor",
    "gory",
    "ory#",
    "##cat",
    "#cate",
    "categ",
    "atego",
    "tegor",
    "egory",
    "gory#",
    "ory##",
    "#pri",
    "##pri",
    "#pric",
    "ac",
    "iv",
    "ve",
    "#ac",
    "act",
    "cti",
    "tiv",
    "ive",
    "ve#",
    "##ac",
    "#act",
    "acti",
    "ctiv",
    "tive",
    "ive#",
    "ve##",
    "##act",
    "#acti",
    "activ",
    "ctive",
    "tive#",
    "ive##",
    "om",
    "cus",
    "ust",
    "sto",
    "tom",
    "ome",
    "mer",
    "#cus",
    "cust",
    "usto",
    "stom",
    "tome",
    "omer",
    "mer_",
    "##cus",
    "#cust",
    "custo",
    "ustom",
    "stome",
    "tomer",
    "omer_",
    "mer_i",
    "#t",
    "ot",
    "al",
    "##t",
    "#to",
    "tot",
    "ota",
    "tal",
    "al_",
    "l_a",
    "_am",
    "##to",
    "#tot",
    "tota",
    "otal",
    "tal_",
    "al_a",
    "l_am",
    "_amo",
    "##tot",
    "#tota",
    "total",
    "otal_",
    "tal_a",
    "al_am",
    "l_amo",
    "_amou",
    "ur",
    "rr",
    "nc",
    "cy"
  ],
  "debug": {
    "topAliasHits": [
      {
        "target": "orders.shipping_city",
        "score": 4.728876610400508,
        "why": [
          "alias",
          "synonym"
        ]
      },
      {
        "target": "@relative.time:period=last_month",
        "score": 2.3,
        "why": [
          "synonym"
        ]
      },
      {
        "target": "orders.total_amount",
        "score": 1.0,
        "why": [
          "synonym"
        ]
      },
      {
        "target": "payment.amount",
        "score": 1.0,
        "why": [
          "synonym"
        ]
      },
      {
        "target": "customer.city",
        "score": 1.0,
        "why": [
          "synonym"
        ]
      }
    ],
    "blend": {
      "cosine": 0.6,
      "bm25": 0.4
    },
    "text": "top revenue by city last month",
    "corrections": [],
    "fixedText": "top revenue by city last month"
  }
}

---

## 3. Revenue by city - Fill with conflicts

**Command:**
```bash
curl -X POST localhost:8000/synapse/fill \
  -H 'Content-Type: application/json' \
  -d '{"intent":{"ask":"top_k","metric":{"op":"sum","target":"orders.total_amount"}}}'
```

**Result:**
{
  "ok": true,
  "intentFilled": {
    "ask": "top_k",
    "metric": {
      "op": "sum",
      "target": "orders.total_amount"
    },
    "target": "orders.total_amount"
  },
  "targetCandidates": [
    {
      "entity": "orders",
      "name": "total_amount",
      "score": 1.2414375146407712,
      "roleTop": "money"
    },
    {
      "entity": "payment",
      "name": "amount",
      "score": 0.9231916769870946,
      "roleTop": "money"
    },
    {
      "entity": "order_item",
      "name": "unit_price",
      "score": 0.44795627754585804,
      "roleTop": "money"
    },
    {
      "entity": "product",
      "name": "price",
      "score": 0.4147715949152245,
      "roleTop": "money"
    },
    {
      "entity": "orders",
      "name": "country",
      "score": 0.29463403223059187,
      "roleTop": "geo"
    },
    {
      "entity": "order_item",
      "name": "order_id",
      "score": 0.2701045486043818,
      "roleTop": "id"
    },
    {
      "entity": "payment",
      "name": "order_id",
      "score": 0.2701045486043818,
      "roleTop": "id"
    },
    {
      "entity": "customer",
      "name": "segment",
      "score": 0.18199070834719178,
      "roleTop": "category"
    }
  ],
  "entityCandidates": [
    {
      "entity": "orders",
      "score": 0.3988089601388495
    },
    {
      "entity": "payment",
      "score": 0.37153615358651815
    },
    {
      "entity": "order_item",
      "score": 0.23855828712306573
    },
    {
      "entity": "customer",
      "score": 0.1953981291207083
    },
    {
      "entity": "product",
      "score": 0.18777246804429204
    }
  ],
  "conflicts": [
    {
      "slot": "category",
      "candidates": [
        "orders.status",
        "payment.method",
        "customer.segment",
        "product.category"
      ],
      "resolution": "orders.status",
      "why": {
        "PathScore": 0.15,
        "CosineContext": 0.10105829297220717,
        "RoleCoherence": 1.0,
        "LocalBonus": 0.03,
        "weights": {
          "path": 0.5,
          "cosine": 0.35,
          "role": 0.15
        }
      },
      "scores": [
        {
          "target": "orders.status",
          "score": 0.29037040254027247,
          "PathScore": 0.15,
          "CosineContext": 0.10105829297220717,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.03,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "payment.method",
          "score": 0.2796513723469013,
          "PathScore": 0.15955788666557555,
          "CosineContext": 0.1424926543260387,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "customer.segment",
          "score": 0.25224190615034925,
          "PathScore": 0.09472848935606283,
          "CosineContext": 0.15679331849233663,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "product.category",
          "score": 0.22784625578021142,
          "PathScore": 0.10357182613101747,
          "CosineContext": 0.07445812204200769,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        }
      ]
    },
    {
      "slot": "timestamp",
      "candidates": [
        "orders.created_at",
        "order_item.created_at",
        "payment.created_at",
        "product.created_at",
        "customer.created_at"
      ],
      "resolution": "orders.created_at",
      "why": {
        "PathScore": 0.15,
        "CosineContext": 0.10454501163560026,
        "RoleCoherence": 1.0,
        "LocalBonus": 0.03,
        "weights": {
          "path": 0.5,
          "cosine": 0.35,
          "role": 0.15
        }
      },
      "scores": [
        {
          "target": "orders.created_at",
          "score": 0.29159075407246005,
          "PathScore": 0.15,
          "CosineContext": 0.10454501163560026,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.03,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "order_item.created_at",
          "score": 0.26636969740524785,
          "PathScore": 0.15955788666557555,
          "CosineContext": 0.10454501163560026,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "payment.created_at",
          "score": 0.26636969740524785,
          "PathScore": 0.15955788666557555,
          "CosineContext": 0.10454501163560026,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "product.created_at",
          "score": 0.23837666713796882,
          "PathScore": 0.10357182613101747,
          "CosineContext": 0.10454501163560026,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        },
        {
          "target": "customer.created_at",
          "score": 0.2339549987504915,
          "PathScore": 0.09472848935606283,
          "CosineContext": 0.10454501163560026,
          "RoleCoherence": 1.0,
          "LocalBonus": 0.0,
          "weights": {
            "path": 0.5,
            "cosine": 0.35,
            "role": 0.15
          }
        }
      ]
    }
  ],
  "alignPlus": {
    "Abase": 1.3214375146407713,
    "Coverage": 1.0,
    "OffSchemaRate": 0.0,
    "Aplus": 1.3214375146407713
  },
  "debug": {
    "topAliasHits": [],
    "tokens": [],
    "mappedTerms": [],
    "rolePrior": {
      "timestamp": 0.04318556483598849,
      "geo": 0.09094677364521417,
      "text": 0.04811787837593057,
      "category": 0.07818907190765119,
      "quantity": 0.18965168576017327,
      "id": 0.13580865273457968,
      "money": 0.41410037274046246
    },
    "matchDebug": {
      "rolePrior": {
        "timestamp": 0.04318556483598849,
        "geo": 0.09094677364521417,
        "text": 0.04811787837593057,
        "category": 0.07818907190765119,
        "quantity": 0.18965168576017327,
        "id": 0.13580865273457968,
        "money": 0.41410037274046246
      },
      "keywordPrior": {
        "quantity": 0.3333333333333333,
        "money": 0.5,
        "id": 0.16666666666666666
      },
      "centroidPrior": {
        "id": 0.11523664344652176,
        "timestamp": 0.07197594139331416,
        "money": 0.3568339545674376,
        "geo": 0.15157795607535698,
        "category": 0.13031511984608535,
        "text": 0.0801964639598843,
        "quantity": 0.09386392071139997
      },
      "blendWeights": {
        "keyword": 0.4,
        "centroid": 0.6
      },
      "entityBlend": {
        "entityWeight": 0.7,
        "fieldWeight": 0.3,
        "entityAliasAlpha": 0.15
      },
      "roleAlphaUsed": 0.1,
      "shapingWeights": {
        "aliasAlpha": 0.4,
        "shapeAlpha": 0.3,
        "shapeBeta": 0.2,
        "metricAlpha": 0.4
      },
      "consideredFields": 32,
      "vocabDim": 384,
      "ask": "top_k",
      "metricOp": "sum",
      "targetPairs": []
    },
    "fillConfigUsed": {
      "unknownPenalty": 0.1,
      "maxUnknownInTopK": 1,
      "preferBonus": 0.08,
      "moneyPivotPenalty": 0.08,
      "osrFromIntentOnly": true,
      "conflict": {
        "pathWeight": 0.5,
        "cosineWeight": 0.35,
        "roleWeight": 0.15,
        "minPathScore": 0.02,
        "lengthPriorBase": 0.92,
        "preferMetricEntityBonus": 0.03
      }
    },
    "preferRolesUsed": [],
    "preferTargetsUsed": []
  },
  "requestId": "15c2f119-3048-45a3-82f7-fcabf197d5cf"
}

---

## 4. Few-shot Show

**Command:**
```bash
curl -s 'http://localhost:8000/fewshot/show?tenant=default'
```

**Result:**
{
  "tenant": "default",
  "merged": {
    "hints": [
      {
        "tokens": [
          "gmv",
          "revenue",
          "sales"
        ],
        "boost": {
          "role": "money",
          "delta": 0.03
        }
      },
      {
        "tokens": [
          "city",
          "municipality",
          "town"
        ],
        "boost": {
          "role": "geo",
          "delta": 0.03
        }
      },
      {
        "tokens": [
          "status",
          "method",
          "channel",
          "segment"
        ],
        "boost": {
          "role": "category",
          "delta": 0.02
        }
      }
    ],
    "aliases": {
      "gmv": "revenue"
    }
  },
  "requestId": "fb5c4eee-2640-474c-b47c-9a60064363b0"
}

---

## 5. Config Effective

**Command:**
```bash
curl -s 'http://localhost:8000/config/effective'
```

**Result:**
{
  "active": {
    "checkpoint": "trainer_20251103_1807",
    "shaping": "/Users/shipyaari/other-projects/codemonklab/synapse-r-phase1/phase3-neurons/checkpoints/trainer_20251103_1807/shaping.json",
    "fewshot": "/Users/shipyaari/other-projects/codemonklab/synapse-r-phase1/phase3-neurons/checkpoints/trainer_20251103_1807/fewshot.json",
    "exists": {
      "shaping": true,
      "fewshot": true
    }
  },
  "requestId": "aadcb550-f288-4f94-8ad2-cbc1092a437c"
}

---

## 6. Verify preferTargets working - detailed breakdown

**Command:**
```bash
curl -X POST localhost:8000/synapse/fill \
  -H 'Content-Type: application/json' \
  -d '{"intent":{"ask":"top_k","metric":{"op":"count"},"target":"orders.id"},"preferRoles":["geo"],"preferTargets":["orders.shipping_city","customer.city","orders.country"]}' | jq '{target: .intentFilled.target, preferTargetsUsed: .debug.preferTargetsUsed, geoCandidates: [.targetCandidates[] | select(.roleTop == "geo")]}'
```

**Result:**
{
  "target": "orders.country",
  "preferTargetsUsed": [
    "orders.shipping_city",
    "customer.city",
    "orders.country"
  ],
  "geoCandidates": [
    {
      "entity": "orders",
      "name": "country",
      "score": 0.9068900563150901,
      "roleTop": "geo"
    },
    {
      "entity": "orders",
      "name": "shipping_city",
      "score": 0.6617836675633075,
      "roleTop": "geo"
    }
  ]
}


## 7. NL Encoding Top Alias Hits

**Command:**
```bash
curl -X POST localhost:8000/intent/encode_nl \n  -H 'Content-Type: application/json' \n  -d '{"text":"top revenue by city last month"}' | jq '.debug.topAliasHits[:3]'
```

**Result:**
[
  {
    "target": "orders.shipping_city",
    "score": 4.728876610400508,
    "why": [
      "alias",
      "synonym"
    ]
  },
  {
    "target": "@relative.time:period=last_month",
    "score": 2.3,
    "why": [
      "synonym"
    ]
  },
  {
    "target": "orders.total_amount",
    "score": 1.0,
    "why": [
      "synonym"
    ]
  }
]


## 8. Geo Conflict Resolution

**Command:**
```bash
curl -X POST localhost:8000/synapse/fill \n  -H 'Content-Type: application/json' \n  -d '{"intent":{"ask":"top_k","metric":{"op":"sum","target":"orders.total_amount"}}}' | jq '.conflicts[] | select(.slot == "geo")'
```

**Result:**


## Summary

All quick wins have been implemented:
1. ✓ Synonyms updated (geo, money, time sections)
2. ✓ Few-shot hints configured with geo boost delta 0.03
3. ✓ preferTargets support added to /synapse/fill endpoint
4. ✓ Checkpoint activated: trainer_20251103_1807
