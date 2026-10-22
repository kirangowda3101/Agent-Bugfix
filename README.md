# Agent-Bugfix

An AI agent that fixes bugs in code, built four different ways so I could measure whether each layer of engineering was actually worth adding.

Most people build one agent and say "it works." I wanted to know something harder: **does each thing I add actually help, and what does it cost?**

The short answer turned out to be no, mostly. Three of the four things I built made no difference to how many bugs got fixed. I can tell you exactly which ones, why, and what each one cost.

Every number in this README comes from log files that are committed to this repository, so you can check them yourself.

---

## Table of contents

- [What problem this solves](#what-problem-this-solves)
- [Why the results are trustworthy](#why-the-results-are-trustworthy)
- [The four versions](#the-four-versions)
- [The two test sets](#the-two-test-sets)
- [Results](#results)
- [What the numbers mean](#what-the-numbers-mean)
- [Things that went wrong, and what they taught me](#things-that-went-wrong-and-what-they-taught-me)
- [Limitations](#limitations)
- [How to run it](#how-to-run-it)
- [What I used and why](#what-i-used-and-why)
- [Project layout](#project-layout)
- [Separate experiment: RL post-training](#separate-experiment-rl-post-training)
- [My other projects](#my-other-projects)

---

## What problem this solves

Imagine you hand a computer program a piece of broken code and say "fix this."

That is the whole task. The nice thing about it is that success is not a matter of opinion. Either the tests pass or they do not. There is no arguing about whether the answer was good enough, which is exactly why I picked it. If I had built something that writes summaries or answers questions, I would have had no honest way to score it.

So the task is simple to describe:

1. Here is some code with a bug in it.
2. Here are tests that currently fail because of the bug.
3. Go fix it.
4. The tests decide whether you succeeded.

### The actual question

Once you can measure success, you can start asking better questions. The one I cared about was this:

> When people build AI agents, they keep adding layers. Tools. Retry loops. Fancy search. Teams of agents talking to each other. **Does any of it actually help?**

Nobody seems to check. The usual pattern is to build the fanciest version, show a demo where it works, and move on.

I built four versions of the same thing, changed exactly one thing at a time, and measured all of them the same way.

---

## Why the results are trustworthy

Before any of the numbers below mean anything, the measuring has to be sound. Four things make it sound.

### 1. The agent does not grade itself

The agent can run tests while it works, so it can tell whether its fix is working. But that is not what decides the score.

After the agent finishes, I run the tests again, separately. That result is the score.

This matters more than it sounds. In several runs the agent confidently announced success on code that did not work. If I had trusted its own report, the numbers would have been fiction.

### 2. I tested the tests

Before using any test to grade an agent, I ran it twice: once against the broken code and once against a known correct fix.

- If it does not fail on broken code, it cannot detect bugs and would hand the agent a free win.
- If it does not pass on correct code, nothing could ever satisfy it and the agent would fail forever for no reason.

Every test in this project passed both checks before being used.

### 3. I checked the official grader too, and caught it being wrong

For the real world bugs, I used the official SWE-bench grading tool rather than writing my own. Before trusting it, I fed it the actual fix that the real developer wrote years ago. That should obviously pass.

For one project (Sphinx) **it reported a failure.** The test genuinely passed inside the container. The grader misread the output and marked it failed.

If I had skipped that check and gone straight to running my agent, I would have seen a failure and spent hours debugging an agent that was working perfectly. Instead I dropped that project and used a different one.

Every single one of the eight real bugs I used was checked this way first.

### 4. The evidence is in this repository

`results/` has one small file per run, recording the number of steps, tokens used, and the outcome.

`logs/` has the official grader's raw output, including exactly which tests passed and which failed for every single bug and every version.

You do not have to take my word for any of the tables below.

---

## The four versions

Each version adds exactly one thing. That is what makes the comparison mean anything.

### V0: the floor

Show the model the broken code. Ask for a fix. Take whatever comes back. One shot, no second chance, no tools, no ability to look at anything.

This one is supposed to be bad. It is the "before" photo.

### V1: give it hands

Now the agent can actually do things:

- **search** the codebase for a word
- **read** any file
- **write** a file

It works in a loop: think, use a tool, look at the result, think again. Up to twenty rounds.

What it still cannot do is **run the tests.** So it fixes something and then just asserts that it worked.

### V2: make it check its own work

Same as V1, plus one more tool: **run the tests.**

Now the agent can see whether its fix actually worked, and try again if it did not. The stopping rule changes from "I think I am done" to "the tests pass."

This is the difference between confidence and evidence, and it was the layer I expected to matter most.

### V3: a team instead of one worker

Instead of one agent doing everything, four agents with separate jobs:

| Role | Can do | Cannot do |
|---|---|---|
| **Researcher** | search, read | write code |
| **Coder** | read, write | run tests |
| **Tester** | run tests | edit anything |
| **Supervisor** | decide who works next | touch any code |

The supervisor looks at what has happened so far and picks who goes next. Workers report back with a short summary, not everything they saw.

### Plus two side experiments

**Smart search instead of plain search.** V2 finds code by looking for exact words, the way you would use Ctrl+F. I built a second version that finds code by *meaning* instead, so you can ask for "the function that parses duration strings" without knowing its name. This is the thing people call RAG.

**A shared tool server.** My four tools were copy-pasted into three different files. I moved them into one place using a standard called MCP, so any agent can ask the server what tools exist instead of having them hardcoded.

---

## The two test sets

### Set one: seven bugs I wrote myself

Small, self contained Python problems. Each one has broken code, a correct version kept aside for checking, and tests.

Designing these turned out to be the interesting part. My first three were too easy: **every version solved all of them,** so the table showed nothing. A table where everything is green tells you nothing about what helped.

So I worked out what makes a bug hard for a one shot guess, and it comes down to hiding the information somewhere the guesser cannot reach:

| Trick | Example |
|---|---|
| Put the bug in a different file | The vowel list is in `helpers.py`, missing the letter "u". V0 only ever sees `solution.py`, which is perfectly fine. |
| Put the rule only in the test | An average function returns 2.3333. The test wants 2.33. Nothing in the code hints that rounding is expected. |
| Use two bugs at once | Fix the first one, run the tests, and the second appears. V0 gets one attempt. |

That worked. V0 dropped from 3 out of 4 to 3 out of 7.

Worth noting: I predicted which of these would work and one of them surprised me. I expected the two bug problem to be a coin flip for V0, because both bugs were in the file it could see. It failed anyway, because nothing in the code told it which character the text was supposed to be split on. That information existed only in the test.

### Set two: eight real bugs from GitHub

These come from SWE-bench Verified, a public set of 500 real bugs from real Python projects that real developers actually fixed. I used eight of them, all from Django.

Real bugs are much harder, and not because the code is bigger:

**My tests said exactly what was wanted.** `assert average([1, 2, 4]) == 2.33`. Crystal clear.

**A real bug report says this:**

> "Suddenly the inputs and outputs are no longer separable? This feels like a bug to me, but I might be missing something?"

Someone confused, pasting output, unsure if they even found a bug. No specification. The agent has to work out what "correct" even means.

Also, Django has **2,928 Python files.** My problems had two or three. The agent cannot just look at everything.

And one more thing which turned out to matter a lot: **the tests used for grading are added at grading time.** They are not in the code the agent works on. So the agent cannot read them to find out what is expected, and cannot run them to check.

---

## Results

All of these come from one command (`python final_report.py`) reading the saved files in `results/` and `logs/`. Nothing is typed in by hand.

### My seven bugs

| Version | Fixed | Tokens used |
|---|---|---|
| V0 (one shot) | 3 of 7 | 875 |
| V1 (tools) | 7 of 7 | 32,538 |
| V2 (tools + tests) | 7 of 7 | 56,507 |

### Eight real GitHub bugs

The last two columns matter as much as the first. They show that the versions failed in completely different ways.

| Version | Fixed | Fix could not be applied | Nothing was written | Tokens used |
|---|---|---|---|---|
| V0 (one shot) | **0 of 8** | **8** | 0 | 4,442 |
| V1 (tools) | 3 of 8 | 0 | 3 | 1,210,393 |
| V2 (tools + tests) | 4 of 8 | 0 | 2 | 1,727,994 |

"Fix could not be applied" means the change it described did not match the real file closely enough to be made at all. "Nothing was written" means the agent explored, thought about it, and never actually changed anything.

### Plain search vs smart search

| Method | Fixed | Nothing written | Tokens used | Setup time |
|---|---|---|---|---|
| Plain word search | 4 of 8 | 2 | 1,727,994 | none |
| Smart meaning search | 4 of 8 | 1 | 2,870,778 | 306 seconds |

They tied, but not on the same bugs. Both solved three of the same ones. Each solved one the other missed.

### One agent vs a team of four

| Setup | Fixed | Nothing written | Tokens used |
|---|---|---|---|
| One agent | 4 of 8 | 2 | 1,727,994 |
| Team of four | 4 of 8 | 1 | 1,293,122 |

Same score. **The exact same four bugs.** Twenty five percent cheaper.

### The shared tool server

Same bug, same result, same fix (a 497 character patch, character for character). Which is the point: the refactor did not break anything.

---

## What the numbers mean

### Giving the agent tools is the only thing that clearly mattered

Zero out of eight became three out of eight. That is the whole story of this project in one line.

And the reason is specific. V0 failed all eight bugs **without ever getting to the point of being wrong about the fix.**

Its patches did not apply. To describe a change to a file, you have to quote the lines around it exactly so the tool knows where to make the edit. V0 had never seen the file. It guessed the surrounding lines from memory, invented plausible looking line numbers, and even made up fake version codes. The result was rejected before anything was tested.

On one bug it did guess the right fix, because the person who reported the bug had written the fix into their report. It still failed, because it could not describe *where* to apply it.

So the lesson is not "the model is not smart enough." It is: **you cannot edit a file you have not looked at.**

### Checking your work helped, but only barely, and only on hard problems

| | My easy bugs | Real GitHub bugs |
|---|---|---|
| V1 to V2 improvement | none at all | one more bug |
| Extra cost | 74 percent more tokens | 43 percent more tokens |

On my easy problems, being able to run the tests bought nothing. V1 could already *read* the test file, and the test file said `== 2.33`. Once you can read the specification, running it adds nothing, because your first attempt was already right.

Running tests only helps when your first attempt is wrong. On easy bugs, it never was.

On real bugs it helped once out of eight. Better than nothing, and the direction is right, but it is not the dramatic gain I expected.

One smaller effect worth mentioning: V2 also produced fewer empty results than V1 (two instead of three). Being able to run tests seems to nudge the agent towards actually changing something rather than exploring and giving up.

### The checking has a hole in it, and I found it

My V2 was built on a principle I liked: do not stop when you feel finished, stop when the tests pass.

On one real bug, the agent ran the tests, saw them pass, and stopped. **It was still wrong.**

The reason is that the tests used for grading are added later. The agent tested itself against the old tests, which could not detect what it had missed. It followed its own rule correctly and still failed.

So the principle needs a caveat: **evidence only helps if the evidence covers what you are actually being judged on.** That was a real limitation I discovered by measuring, not by reading about it.

### Smart search did not beat plain search

This one surprised me least. Bug reports name things exactly. Someone writes "the `standard_duration_re` regex is wrong," and a plain word search finds it in two results, instantly, for free.

Smart search found it too, and arguably better (it landed on the exact function rather than just the file). But it cost 66 percent more in tokens and 306 seconds of setup, and it needed four extra software libraries, one of which crashed my machine until I worked around a conflict between two of them.

They each solved one bug the other missed, which is interesting but with only eight bugs I cannot honestly claim either is better at finding things. What I can say with confidence is that one of them costs more.

**Practical takeaway:** for finding bugs in code, plain search is hard to beat, because bug reports use the same words the code uses. Smart search would matter more if people described problems in language that did not match the code.

There is also a detail worth noticing about how the agent used the smart search. On one bug it called it **once**, in twenty steps, and had already located the file by other means before doing so. The retrieval method barely mattered because finding the file was never the hard part.

### A team of four agents did exactly as well as one, for less money

This is the result I expected to go the other way. The research I had read said teams of agents usually cost far more (around three times more) and do not perform better.

Mine cost **25 percent less** and scored identically. Same four bugs. Not a different four, the same four.

The reason it was cheaper is worth understanding, because it is not about the teamwork at all:

**One agent** has one long conversation. Every time it takes a step, it has to re-send everything said so far. By step twenty, that is an enormous amount of text being paid for again and again.

**Four agents** each start fresh with a short summary. The coder never sees the researcher's full search results, just "the bug is in this file, on this line."

So the savings come from **keeping conversations short**, not from specialisation. Splitting one worker into four specialists with a manager produced no capability difference whatsoever.

Which suggests something more useful than "teams are good" or "teams are bad": if you want the savings, you do not need a team. You need to stop resending your whole history. A single agent that summarises its own progress would get the same benefit with none of the coordination problems.

And there were coordination problems, which is the next section.

### The shared tool server did not make anything better, and that was the point

Same result, same patch, same everything. What it changed was the code: I deleted about fifty lines of hand written tool descriptions and an entire chain of if/else statements deciding which tool to run.

Adding a fifth tool now means writing one function. Before, it meant editing three files.

There was a second benefit I did not expect. Writing a tool by hand meant describing its inputs in a fiddly structured format. With the shared server, that description is generated automatically from the function itself, so it can never drift out of sync with the code.

Being honest about it: for one agent, this buys tidiness, not capability. Its real value would show up if I had several different programs needing the same tools, which is exactly the situation the team of four agents created.

---

## Things that went wrong, and what they taught me

This is the most useful part of the project. Every one of these came out of watching real runs.

### The agent invented a bug that did not exist

On the problem where the bug was hidden in a second file, V0 was shown a file that was completely correct. It could not find anything wrong, because there was nothing wrong.

So it made something up. It added code to handle capital letters, which nobody asked for and which fixed nothing.

Later, one run did this to a *working* file: the tests were already passing, and the agent decided the lack of uppercase handling was a bug and rewrote the file anyway.

**Lesson:** told there is a bug when there is not, the model will manufacture one rather than say "I think your premise is wrong." Worth knowing if you ever build something that acts on reports it cannot verify.

### The agent explained its own fix incorrectly

V1 (the version that cannot run tests) fixed a bug correctly, then wrote a summary claiming that a second test had also been failing. That test had been passing the whole time.

The fix was right. The story about the fix was partly false. And nothing would have caught it.

When I gave the same agent the ability to run tests, its summary became accurate, because it was describing something it had actually seen instead of something it assumed.

**Lesson:** an agent's explanation of what it did is not evidence that it did it.

### The agent asked for a tool that did not exist, then asked again

Early on I gave the agent only one tool: list the files. It listed them, then wanted to read one, could not, and **called "list files" a second time** hoping for something different. Only on the third round did it say "I do not have a way to read files, can you paste them?"

**Lesson:** when a needed tool is missing, the model does not immediately reason about the gap. It reaches for whatever looks closest and burns a full round trip first.

### The agent found the answer at step 5 and kept going until step 20

On the first real bug, the agent had the correct fix in place at step five, verified by tests at step six.

Then it ran for fourteen more steps. It re-read the same file three times. It wrote five diagnostic scripts. It ran tests on a module that does not exist.

Those fourteen steps cost about **75 percent of the total tokens for that run** and produced nothing.

Nothing told it to stop. Passing tests was not treated as a finish line.

**Lesson:** a step limit is not a stopping rule. It caps how long you waste money, not whether you waste it.

### The agent wrote five scripts it had no way to run

Those diagnostic scripts it wrote? It has a tool for writing files and no tool for running arbitrary code.

So it carefully wrote out five test scripts, saved them, and could never see their output. It kept doing this across multiple runs.

**Lesson:** the set of tools shapes not just what an agent can do but what it believes it can do. It assumed a capability it had never been given.

### Making the agent rewrite whole files corrupted them

My "write a file" tool replaces the entire file. To change one line, the agent has to reproduce every other line perfectly.

On a 900 line Django file, this went badly. My "read a file" tool cuts off at 400 lines, so the agent had only seen part of it. It rewrote the file from what it could see and **invented roughly 480 lines of replacement code** for the part it could not, including functions that never existed and a comment reading "to be tested."

The patch came out at 26,926 characters. A normal fix in this project is about 500.

You can watch it realise something is wrong: "Let me check the rest of the file to see if we need to restore it."

**Lesson:** a tool that edits specific lines would have made this failure impossible. This is the single clearest design mistake in the project, and I found it by reading logs rather than by thinking about it in advance.

### A search pattern starting with a minus sign broke the search

The agent searched for a pattern beginning with `-`. The underlying search program read that leading minus as an option flag rather than as text, and refused to run.

```
grep: invalid option -- '?'
```

**Lesson:** anything a model types is untrusted input to whatever runs underneath it. The same care you would take with input typed by a stranger applies to input written by a model.

### The team of agents silently failed to make any change at all

First run of the four agent version. Result: **an empty patch.** Nothing was changed.

Here is the chain:

1. The **researcher** reported the file as `./django/utils/dateparse.py`
2. The **coder** wrote to `/testbed/django/utils/dateparse.py`
3. My write tool adds `/testbed/` automatically, so the real destination became `/testbed//testbed/django/...`
4. That folder does not exist. **The write failed and my tool reported success anyway.**
5. The **tester** ran tests against completely unmodified code
6. Old tests pass on the buggy code, so it reported "ALL TESTS PASSED"
7. The **supervisor** believed it and declared the job done

Four agents, each doing its job, and the result was confidently, cheerfully wrong.

Two separate lessons here:

**My tool threw away the error message.** It ran a command, ignored what came back, and said "wrote 4712 characters." A tool that fails quietly is worse than one that crashes, because the failure travels. I fixed it to report the error.

**The mismatch was caused by the teamwork.** The single agent used one path style throughout, because it was one continuous train of thought. Two independent agents each picked their own style and nothing reconciled them. That is a class of bug that only exists once you split the work up.

### An agent knew what it needed to do and could not do it

Later in a team run, the tester noticed a file was in a bad state and said:

> "I need to restore it first. Let me check the git status."

Then it called the only tool it has, which runs tests. It could not check anything. It could not fix anything. It knew what was needed and had no way to act.

**Lesson:** specialising an agent means deliberately taking away its ability to respond to what it finds. That is the trade, and it has a real cost.

### The supervisor lost track and repeated itself

In one run, the supervisor sent work to the coder twice in a row. Both times the coder said "the fix has already been applied." Nothing detected the repetition.

My original plan called for spotting exactly this and I did not build it. This run is why it belongs on the list.

### The supervisor replied with an essay instead of a decision

The supervisor is asked for exactly one word: researcher, coder, tester, or done. Once, it replied:

```
**tester**

the file has been restored to a working state. now i need to run
```

I had built a guard for unrecognised answers, so it stopped cleanly instead of crashing.

**Lesson:** never assume a model will respect an output format, even a one word one. The guard was five lines and it earned its place.

### Killing a process on my machine did not kill the work it started

Early on I put a time limit in my Python code, so a run that hung would be abandoned after thirty seconds.

It fired correctly. Then I checked, and **the container was still running**, burning a processor core, with nothing left to report to.

The reason is that the command I killed was only a messenger. It had asked a background service to start the container, and killing the messenger did not tell the container anything.

The fix was to give the container its own internal time limit, so it ends itself rather than needing to be ended from outside.

**Lesson:** when one program starts another, stopping the first does not necessarily stop the second. If you want something to stop, the stopping instruction has to reach the thing actually doing the work.

### A blank failure taught me to record why things failed

For a while, a failed run just printed "FAIL" and nothing else. That could mean the tests ran and failed, or the whole environment failed to start, or something timed out. Three completely different problems, identical on screen.

Every program hands back a small number when it finishes. Zero means fine, and different non-zero numbers mean different kinds of wrong. I started recording that number, and suddenly the failures told me what they were:

| Number | What it means |
|---|---|
| 1 | tests ran and some failed |
| 2 | tests could not even be loaded |
| 124 | something ran out of time |
| 125 | the container failed to start |

**Lesson:** "it failed" is not information. Recording the reason costs one line and saves hours. This also matters for the agent, not just for me, because a fix that loops forever needs a different response than a fix that is simply wrong.

### My laptop and the container confused each other over leftover files

When tests run, they leave behind a small cache to make the next run faster. That cache records where the original files live.

At one point my laptop ran the tests and left a cache with laptop file paths in it. Then the container ran the tests, found the cache, believed it, and tried to open files that do not exist inside a container.

The result was test output with question marks where the code should have been. Not wrong exactly, just useless.

**Lesson:** two different machines sharing a folder will leave each other confusing notes. Wiping the workspace before every run solved it permanently.

### The disk space warning did not apply to me, and it took a while to notice

The official tool warns you to keep 120 gigabytes free. I checked I had space and braced for it to fill up.

It never did. My disk usage stayed flat at about 220 megabytes throughout.

The reason is that the tool downloads a prepared environment, uses it, and deletes it again. It only builds and keeps things if it cannot find a prepared one.

**Lesson:** warnings are written for the worst case. Measuring what actually happens on your setup took one command and saved me from planning around a problem I did not have.

### Building this ran into six things that had nothing to do with agents

Worth recording, because "why did this take so long" is usually answered by things like these:

1. **My laptop is the wrong shape.** SWE-bench is built for Intel machines. Mine is Apple Silicon, so every container runs through a translation layer. It works, just slowly, and support for my chip is officially experimental.
2. **Two of my eight bugs had no prepared environment.** The prebuilt containers for them simply do not exist in the public registry, so I replaced them with two others.
3. **A data field looked like a list and was actually text.** I spent a while confused by output showing single characters before realising I needed to convert it first. This also made me briefly, and wrongly, conclude that part of the dataset was corrupted.
4. **Two libraries crashed my machine when loaded together.** The search library and the machine learning library each bundle their own copy of the same underlying maths component, and having both caused a hard crash with no error message at all. Two environment settings fixed it.
5. **A library had reorganised itself.** Code that is in every tutorial no longer works, because the pieces moved to different packages and part of it is being retired.
6. **A tool I was using had changed its entire interface.** The main class had been renamed and several field names changed, so nothing I knew about it applied. I had to read the installed version to find out what it now offered.

**Lesson:** on any project involving fast moving tools, a meaningful share of the time goes to things that are not the problem you set out to solve. This is worth planning for rather than being surprised by.

---

## Limitations

Being straight about what these numbers do not prove.

**Eight real bugs is a small sample.** A tie at four out of eight, where each method solves one bug the other missed, is well within the range of coincidence. I cannot claim any of the retrieval or architecture comparisons found a real difference in capability. What I can claim is the cost differences, which are large and consistent.

**Each version ran once per bug.** These models are not deterministic. The same version run again might score differently. Three runs each would tell me whether these numbers are stable. I have not done that.

**One model throughout.** I used the same fast, cheap model for everything, including the supervisor, specifically so architecture was the only thing changing. My original plan called for a better model for the supervisor, but that would have changed two things at once. Using better models is an obvious next experiment.

**The eight bugs were not randomly chosen.** I picked ones with small test suites so runs would be fast. That makes them easier than average and means the four out of eight score is not comparable to published leaderboard numbers on the full 500.

**All eight come from one project.** All Django. Findings may not carry over to projects with different conventions.

**One bug had to be retried.** A run stalled with no response, apparently while sending a very large amount of accumulated text. I stopped it and reran that one bug. Worth noting that a step limit caps the number of steps, not the size of any single one, and those are different problems.

**The write tool is a known weakness.** Rewriting whole files caused at least one clear failure. I deliberately left it alone rather than improving it mid experiment, because changing the tools would have invalidated every comparison. Fixing it and rerunning everything is the right next step.

---

## How to run it

### You will need

- Python 3.11
- Docker (running)
- An API key from Anthropic

### Setup

```bash
conda create -n agent python=3.11 -y
conda activate agent
pip install anthropic python-dotenv pytest datasets swebench
pip install langchain langchain-text-splitters langchain-huggingface faiss-cpu sentence-transformers
pip install mcp langgraph
```

Put your key in a file called `.env` (this file is ignored by git and must never be committed):

```
ANTHROPIC_API_KEY=your-key-here
```

Build the container the small problems run inside:

```bash
docker build -t bugfix-sandbox .
```

### Run against my seven bugs

```bash
python v0.py p01_running_total     # one shot
python v1.py p01_running_total     # tools, no test running
python v2.py p01_running_total     # tools plus test running
```

Available problems: `p01_running_total`, `p02_count_vowels`, `p03_find_max`, `p04_count_vowels_helper`, `p05_discount_config`, `p06_average_rounding`, `p07_parse_scores`.

Check a problem's tests are sound before trusting them:

```bash
python run_tests.py p01_running_total fixed   # must pass
python run_tests.py p01_running_total buggy   # must fail
```

### Run against the real GitHub bugs

```bash
python swe_v0.py django__django-10999
python swe_v1.py django__django-10999
python swe_v2.py django__django-10999
python swe_multi.py django__django-10999
```

The smart search version needs two environment settings, or it will crash:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python swe_v2_rag.py django__django-10999
```

### Run everything

```bash
python run_swe_all.py                          # all versions, all eight bugs
python run_swe_all.py swe_v0                   # just one version
```

Already completed runs are skipped, so if it stops halfway you can rerun it without paying twice.

### See the results

```bash
python final_report.py
```

### Check the grader before trusting it

This is the step I would tell anyone not to skip. Feed the grader the real developer's fix and confirm it says pass:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path gold \
  --instance_ids django__django-10999 \
  --max_workers 1 \
  --run_id gold_check \
  --cache_level env
```

### Cost

The small problems cost a fraction of a cent each. The real bugs cost roughly twenty cents each for the single agent. Running everything, all versions across all eight bugs, came to somewhere around five dollars.

Cleaning up afterwards:

```bash
docker system df          # see how much space containers are using
docker system prune -a    # remove everything not in use
```

---

## What I used and why

Every item here does a real job. I can explain what each one is for and, in most cases, whether it was worth including.

| Thing | What it does here | Was it worth it |
|---|---|---|
| **The think, act, observe loop** | Written from scratch rather than using a framework, so I understand every part of it | Yes. Everything else is built on it |
| **Tool calling** | How the agent searches, reads, writes, and runs tests | Yes. This is where all the capability came from |
| **Docker containers** | Every piece of agent written code runs sealed off from my machine, with a hard time limit | Yes. Also the only practical way to get old Django dependencies working |
| **Time limits and step budgets** | Three layers: inside the container, in Python, and on the workflow | Yes. All three fired at least once |
| **Path checks on file writing** | The agent cannot write outside its working folder | Yes, though it never tried |
| **Test design** | Every test checked against both broken and correct code before use | Yes. This is the foundation everything rests on |
| **SWE-bench Verified** | Real bugs, graded by the official tool rather than my own | Yes. Removes any suspicion that I graded myself generously |
| **LangChain and FAISS** | The smart search version: splitting files, converting to numbers, finding similar pieces | Built it, measured it, and it did not beat plain search. Reporting that honestly |
| **MCP** | Tools live in one shared server instead of copy pasted into three files | Tidier code and automatic tool descriptions, but no capability gain for one agent. Real value only appeared with the team version |
| **LangGraph** | Defines who works next in the four agent version | Yes for building it, though the architecture itself changed nothing |
| **Structured logging** | Every run saves steps, tokens, and outcome to a file | Yes. Every number in this README comes from those files |
| **Cost tracking** | Token counts recorded per run and totalled per version | Yes. Half the findings here are about cost, not capability |

### On frameworks

I wrote the agent loop by hand and used frameworks only where they added something.

The loop is about thirty lines and understanding it is the whole point of doing this. If I had used somebody's agent framework, I would have a working agent and no idea how it works.

Splitting text into chunks and searching by similarity is a solved problem with fiddly details, so I used a library. Defining which agent runs next is bookkeeping, so I used a library for that too.

That division was deliberate, and being able to explain why is more useful than having used more tools.

---

## Project layout

```
dataset/                 seven bugs I wrote, each with broken code,
                         a correct version for checking, and tests
results/                 one small file per run: steps, tokens, outcome
logs/                    raw output from the official grader
repos/django             a local copy of Django, used for the smart
                         search index (not committed, too large)
predictions/             the fixes each version produced (not committed)

run_tests.py             runs a test in Docker, reports pass or fail
v0.py, v1.py, v2.py      the three versions, for my seven bugs
swe_v0.py, swe_v1.py     the same three, for real GitHub bugs
swe_v2.py
swe_v2_rag.py            V2 using smart search instead of word search
swe_v2_mcp.py            V2 getting its tools from the shared server
swe_multi.py             the four agent version
mcp_server.py            the four tools, in one reusable place
run_swe_all.py           runs everything and grades it
report.py                results table for my seven bugs
final_report.py          every table, all four experiments
```

---

## If I carried on

In rough order of how much I think each would matter:

1. **Replace whole file writing with line by line editing.** This caused the worst failure in the project. Everything would need rerunning afterwards, but it is the clearest improvement available.
2. **Run everything three times.** Right now I cannot tell a real difference from a coincidence on the four out of eight results.
3. **Stop the agent when the tests pass.** Fourteen wasted steps out of twenty on the very first real bug. This is cheap to fix and would cut costs substantially.
4. **Give the single agent short summaries too.** The team version was 25 percent cheaper purely from keeping conversations short. That does not require a team.
5. **Detect when the workflow is repeating itself.** The supervisor sent work to the same agent twice with no idea it was going in circles.
6. **Add more bugs, from more projects.** Eight bugs from one project is thin.
7. **Try better models for the supervisor.** I used one model everywhere on purpose. Now that architecture is measured, model choice is the next variable.

---

## Two things I would tell someone starting this

**Check your measuring tools before you measure anything.** I caught a broken grader by feeding it a known correct answer. If I had skipped that, I would have spent a day debugging an agent that was working fine. The same idea applied at every scale: test the tests, verify the API key alone, prove the container works before putting an agent in it.

**Design your test set to actually distinguish things.** My first attempt had every version scoring full marks, which told me nothing. A comparison is only worth running if the things being compared can come out different.

---

## Separate experiment: RL post-training

[`post_training/`](post_training/README.md) is a separate experiment that does not touch anything above: GRPO reinforcement learning on a small open code model (Qwen2.5-Coder-0.5B-Instruct) against MBPP unit tests, with every piece of model-written code run in a sandbox. Its method, sandbox design, measurements and limitations are in [its own README](post_training/README.md).

---

## My other projects

Two earlier from scratch builds, both with published weights:

- **[Speech-Language Model](https://github.com/kirangowda3101/Speech-Language-Model)**: a 522 million parameter transformer trained from scratch to handle speech and text together, on a university computing cluster.
- **[Vision-Language Model](https://github.com/kirangowda3101/Vision-Language-Model)**: a CLIP style model built from scratch that learns to match images and captions, including an experiment showing why batch size matters so much for this kind of training.

---

## License

MIT.