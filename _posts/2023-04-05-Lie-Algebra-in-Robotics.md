---
layout:     post
title:      "Lie Theory for the Roboticist"
subtitle:   "Full of stars"
date:       2023-04-05 01:00:00
author:     "Yubin"
header-img: "img/Headers/mio.jpg"
mathjax: true
catalog: true
tags:
    - Robotics
    - Math
    - Lie Algebra
---

由于涉及大量的模糊描述，并非严谨的证明，这篇文章主体用中文完成，部分术语以英文给出，希望用一个更好理解的方式解释李群的引入和它在机器人学中的应用。本文涉及大量的可视化工作，部分注明引用，如有侵权请联系我删除。

# Lie Theory for the Roboticist

## Why we need Lie theory?

考虑一个三维的旋转矩阵$R$，它具有正交性，且行列式的值为$1$：

