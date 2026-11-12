# Voices of the Future

*Building with modern voice AI models.*

<div class="sv-status">
<p><strong>Draft, under review.</strong></p>
<p>All twelve chapters are drafted and under review, so expect changes. Subscribe to the <a class="reference external" href="https://books.sugiv.fyi/feed.xml">books RSS feed</a> to hear when chapters are revised.</p>
</div>

## What you will build

One project runs through the whole book: a voice receptionist for Halfwheel Cycles, a fictional bike repair shop. It picks up calls, books repairs, checks whether a bike is ready, answers questions, takes messages and screens spam calls. Each chapter adds a working piece, and by the end you have built it, measured it and put it on a phone line.

## What it will cover

- How speech recognition works, from CTC to transducers to LLM decoders, and how to choose a model for your callers
- Listening live: streaming, turn-taking and interruptions
- How modern text-to-speech works, and how to direct a voice's emotion and delivery
- Building a voice agent with tools and guardrails, and screening spam calls
- How speech language models evolved, from Spirit LM to Moshi and CSM
- Evaluating, deploying and scaling voice agents
- Case studies from real projects, and a field guide of common problems and fixes

## Who it is for

Engineers and builders at every level who want to put voice AI models to work in real products. Every chapter has a quick path that works in minutes, an explanation of how it works, and notes on running it in production.

## Running the code

Every chapter runs on your own laptop from saved results in seconds. Most of the models also run on a free Google Colab T4 GPU; a few of the larger ones (Fish Audio's S2-Pro, Moshi, and the 8B language model) need a GPU with more memory, and their chapters say so. I checked every notebook's model steps on a 16 GB GPU with the T4's float16 limits. Each chapter from 2 to 12 has a notebook in the [companion repo](https://github.com/sugix/voices-of-the-future), with the saved results, the simulators and the capture scripts.
