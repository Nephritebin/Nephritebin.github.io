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

之后你就可以看到这样的一张图，注意小括号里面是全局路径，一般保存在博客源文件的 image 文件夹下，我按月份分类保存。

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

如果你的图片插入和 Latex 渲染没问题，那这个博客的功能已经足够了，剩下的就是 Markdown 语法的东西了。如果对语法不是很熟悉的话，可以去参考一下这个 [Markdown Guide](https://www.markdownguide.org/basic-syntax/)，不过只要能够多写自然就会慢慢变得熟练起来了。

## 进阶与前端

一些更进阶的玩法包括在 Markdown 中嵌入 `HTML` 代码，进而修改文字，图片与视频的格式。由于我自己对前端了解也不多，因此这一部分就一边探索一边更新。

### 图片插入进阶

利用下面的代码模板，可以插入带有文字描述的图片，调整图片的大小与位置。可以根据自己喜好修改字体，图片文字距离，字号等等参数。注意图片的路径与 Markdown 中插入时不同。一个很好的消息是我发现 Onedrive 支持导出文件嵌入的代码块用于个人博客，因此我们可以直接使用 Onedrive 中的直链

```html
<figure>
    <img width="80%" align="middle" src="https://onedrive.live.com/embed?resid=F7F75CDEDD2EF73D%21240201&authkey=%21AAogQ6Z8rlJkir0&width=2560&height=1440" style="margin-top: 0px; margin-bottom: 5px"/>
    <div style="font-family: STXingkai, sans-serif; font-size: 12px; text-align: center; margin-top: 0px;">
    甘雨可爱，贴贴 </div>
</figure>
```

效果如下所示

<figure>
    <img width="80%" align="middle" src="https://onedrive.live.com/embed?resid=F7F75CDEDD2EF73D%21240201&authkey=%21AAogQ6Z8rlJkir0&width=2560&height=1440" style="margin-top: 0px; margin-bottom: 5px"/>
    <div style="font-family: STXingkai, sans-serif; font-size: 12px; text-align: center; margin-top: 0px;">
    甘雨可爱，贴贴 </div>
</figure>

当然，如果更喜欢折腾的话，可以更改网站的`css`模版，之后实现各种自定义样式，比如我想实现上面写的那种样式，我们可以在`css`文件夹下自己定义一个模版`personal.css`：

```css
/* For the figure container */
.figure-container {
    width: 100%;            /* You can adjust as per your requirement */
    margin: 5px auto;     /* Centering the figure horizontally */
    position: relative;   
}

/* For the image inside the figure */
.figure-container img {
    width: 100%;
    height: auto;
    display: block;
}

/* For the caption under the image */
.figure-caption {
    text-align: center;    /* Centers the caption text */
    font-size: 0.9em;      /* You can adjust as per your requirement */
    color: #555;           /* Adjust the caption text color */
    margin-top: 0px;      /* Space between the image and the caption */
}
```

之后把它放在`css`文件夹里，再在`head.html`中插入：

```html
<link rel="stylesheet" href="{{ " /css/personal.css" | prepend: site.baseurl }}">
```

现在，我们就可以在 Markdown 文件中使用上面定义的样式了。不过为了简化 Markdown 文件中的图像和标题的插入，我们可以使用 Jekyll 的 Liquid 模板语言创建一个包含标签。在 `_includes` 目录下创建一个新文件，例如 `custom_figure.html`，并添加以下内容：

```html
<div class="figure-container" style="width: {{ include.width | default: '100%' }};">
    <img src="{{ include.src }}" alt="{{ include.alt | default: 'Image' }}">
    <p class="figure-caption">{{ include.caption }}</p>
</div>
```

我们就可以在 Markdown 文件中使用简化的方式插入图片了，甚至还可以指定需要的参数与输入标识。效果如下所示，和上面插入的图片效果是相同的。在此基础上，你可以用`css`指定更多格式的效果，具体可以询问 ChatGPT 。事实上，这些代码都是 ChatGPT 告诉我的。

{% 
    include responsive_image.html 
    src="https://onedrive.live.com/embed?resid=F7F75CDEDD2EF73D%21240201&authkey=%21AAogQ6Z8rlJkir0&width=2560&height=1440" 
    caption="甘雨可爱，贴贴" 
    width="80%" 
%}

美中不足的是，这样快速插入的办法不能实现 `vscode` 的预览，因此我们只能使用 `jekyll` 的预览进行测试了。

### `iframe` 标签与响应式设计

利用原生的`HTML`语言, 我们可以插入 Bilibili 或者 Youtube 的视频，视频下方的分享按钮可以复制嵌入的代码，例如

```html
<iframe 
    width="640" height="360" 
    src="https://www.youtube.com/embed/M41ID8tLAWU" 
    frameborder="0" 
    allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
</iframe>
```

可以插入 Youtube 的视频， 效果如下所示，快来和我一起看胡桃 (˵¯͒ བ¯͒˵)：

<iframe 
    width="640" height="360" 
    src="https://www.youtube.com/embed/M41ID8tLAWU" 
    frameborder="0" 
    allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
</iframe>

为了使 `iframe` 响应式，我们可以使用一种常见的技巧，即使用固定的宽高比的容器。以下是一个例子，设置 `iframe` 的宽高比为 16:9。和上面一样，我们只需要在`css`文件夹中新建或者修改我们自己定义的样式文件就好：

```css
.responsive-iframe-container {
    position: relative;
    overflow: hidden;
    width: 100%;
    padding-top: 56.25%;
}

.responsive-iframe-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: 0;
}
```

上述 CSS 代码首先设置了一个固定的宽高比容器，然后使用绝对定位来填充 `iframe`，使其填满整个容器。这样，无论屏幕的大小如何，`iframe` 都会保持相同的宽高比，并且会随着容器的大小变化而调整大小。这样一来，我们就可以插入一些`iframe`标签支持的文件了。比如，我们可以试着插入一个来自 Onedrive 的 PDF 文件:

```html
<div class="responsive-iframe-container">
<iframe src="https://onedrive.live.com/embed?resid=F7F75CDEDD2EF73D%21240452&authkey=!AIJlFS5mbpKmT0Y&em=2" frameborder="0" scrolling="no"></iframe>
</div>
```

效果如下所示

<div class="responsive-iframe-container">
<iframe src="https://onedrive.live.com/embed?resid=F7F75CDEDD2EF73D%21240452&authkey=!AIJlFS5mbpKmT0Y&em=2" frameborder="0" scrolling="no"></iframe>
</div>

让我们一起愉快地玩耍吧~
