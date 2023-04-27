---
layout:     post
title:      "My First Blog"
subtitle:   " \"Hello World, Hello Blog\""
date:       2022-09-19 12:00:00
author:     "Yubin"
header-img: "img/Headers/head-5-seconds.jpg"
mathjax: true
catalog: true
tags:
    - Jekyll
    - Front end
---

# For the land of the lustrous

> “Yeah It's on. ”

Yubin 的 Blog 就这么开通了。

感谢 Hux 大神提供的 Jekyll 模板，让我自己没事随便写点又不知道放哪的东西终于有了一个呆着的地方。这篇文章记录一下自己的博客开通的过程，也作为对自己首次用 Jekyll 写博客的测试。从 Hexo 折腾到 Jekyll，终于找到了一个大部分功能都支持，自己也用着顺手的东西，希望这能激励自己以后好好输出吧（笑）。

## 安装与下载

首先，根据 Jekyll 官网的[安装教程](https://jekyllrb.com/docs/)，大致分为以下三步：

1. 安装 prerequisite
   
   需要安装 ruby 和 gcc 和 make，具体过程可以参考[这一文章](https://blog.csdn.net/cheng__lu/article/details/88963229)和[这一文章](https://zhuanlan.zhihu.com/p/47935258)，安装完成后可以在终端检查，如果输入以下命令都不会报错的话，就表示安装成功。
   ```bash
    ruby -v
    gem -v
    gcc -v
    make -v
   ```

2. 安装 Jekyll
   
   只需要简单的在命令行输入以下命令即可：
   ```bash
    gem install jekyll bundler
   ```

3. 下载模版文件
   
   在Github上选择一个你喜欢的模版下载到本地，之后执行以下命令：

   ```bash
    jekyll serve
   ```
   如果能够在 `http://127.0.0.1:4000/` 端口看到你的网页，那么恭喜你，环境的配置已经完成了。 如果你遇到报错信息例如`Bundler::GemNotFound`之类的，不要慌，可以尝试采用这个命令，自己在进行以下操作后成功完成环境配置（参考来源于[这个链接](https://stackoverflow.com/questions/17599717/rails-bundlergemnotfound)）：

   ```bash
    bundle update
   ```
   

## 调试与修改

一个模版的文件包括各种JavaScript，css，html 和 Markdown文件，一般来说我们需要定义和修改的东西大部分都在`_config.yml`里面，发布的内容一般用 Markdown 写在 `_post`文件夹里，如果愿意折腾的话也可以修改模版的 html 文件。以下以 Hux 大神提供的 [Jekyll 模板](https://github.com/Huxpro/huxpro.github.io)为例进行说明，大致需要改动的地方可以参考模版的[说明文档](https://github.com/Huxpro/huxpro.github.io/blob/master/_doc/README.zh.md)。

值得注意的是，`Latex` 的支持和 Google Analytics 由于更新，需要自行改动原 `html`代码才能生效，具体的详细内容可以看我的 Github 改动的 commit。

## 测试与例子

首先，我们来测试一下插入图片，Markdown 的命令为：

```markdown
![5 centimeters per second](/img/Notes/2022-09/head-5-seconds.jpg)
```

之后你就可以看到这样的一张图，注意小括号里面是全局路径，一般保存在博客源文件的 image 文件夹下，我按月份分类保存。目前暂时还没有研究出来怎么调整生成的网页中图片的大小和等比例，有知道的可以戳我。

![Violet](/img/Notes/2022-09/head-5-seconds.jpg)

其次，测试一下 Markdown 的数学公式支持，注意要在 Markdown 的头部加入：

```markdown
mathjax: true
```

我们有，方程 $ax^2 + bx+c=0$ 的解是：

$$
x = \frac{-b\pm \sqrt{b^2-4ac}}{2a}
$$

如果你想打一个矩阵，出来的效果是：

$$
A = I = \left [ \begin{matrix}
    1 & 0 & 0\\
    0 & 1 & 0\\
    0 & 0 & 1
\end{matrix} \right ]
$$

如果你的图片插入和 Latex 渲染没问题，那这个博客的功能已经足够了，剩下的就是 Markdown 语法的东西了。如果对语法不是很熟悉的话，可以去参考一下这个 [Markdown Guide](https://www.markdownguide.org/basic-syntax/)，不过只要能够多写自然就会慢慢变得熟练起来了。一些更加进阶的功能包括利用原生的`HTML`语言插入 Bilibili 或者 Youtube 的视频，视频下方的分享按钮可以复制嵌入的代码，例如

```html
<iframe width="640" height="360" src="https://www.youtube.com/embed/M41ID8tLAWU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

可以插入 Youtube 的视频， 效果如下所示，快来和我一起看胡桃 (˵¯͒ བ¯͒˵)：

<iframe width="640" height="360" src="https://www.youtube.com/embed/M41ID8tLAWU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

让我们一起愉快地玩耍吧~
