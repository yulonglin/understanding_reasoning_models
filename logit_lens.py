# %%
# from transformer_utils.low_memory import enable_low_memory_load

# enable_low_memory_load()
# from transformer_utils.low_memory import disable_low_memory_load

# disable_low_memory_load()
import torch

# # Check that MPS is available
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print(
#             "MPS not available because the current PyTorch install was not "
#             "built with MPS enabled."
#         )
#     else:
#         print(
#             "MPS not available because the current MacOS version is not 12.3+ "
#             "and/or you do not have an MPS-enabled device on this machine."
#         )
# mps_device = torch.device("mps")

import transformers

# path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
path = "Qwen/Qwen2.5-Math-1.5B"

model = transformers.AutoModelForCausalLM.from_pretrained(path).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(path)

import gc
import torch


def cleanup_model(model):
    try:
        if hasattr(model, "base_model_prefix") and len(model.base_model_prefix) > 0:
            bm = getattr(model, model.base_model_prefix)
            del bm
    except:
        pass
    del model

    gc.collect()
    # torch.cuda.empty_cache()


import torch


def text_to_input_ids(text):
    toks = tokenizer.encode(text)
    return torch.as_tensor(toks).view(1, -1).cuda()


text = """Please reason step by step within <think> tags, and put your final answer within \boxed{}.

PROBLEM:
How many even integers between 4000 and 7000 have four different digits?

<think>
Alright, so I need to figure out how many even integers between 4000 and 7000 have four different digits. Hmm, okay. Let's break this down step by step.

First off, the number has to be between 4000 and 7000. That means the first digit can't be less than 4 or more than 6, right? Because 7000 is the upper limit, but since we're dealing with four-digit numbers, the thousands place can be 4, 5, or 6. So, the thousands digit has 3 possible choices: 4, 5, or 6.

Next, the number needs to be even. For a number to be even, its last digit has to be even. So, the units digit can be 0, 2, 4, 6, or 8. That gives us 5 possible choices for the last digit. But wait, we also need all four digits to be different. So, whatever digit we choose for the thousands place, it can't be repeated in the units place, and vice versa.

So, let me outline the steps I need to take:

1. Determine the possible choices for the thousands digit: 4, 5, 6.
2. Determine the possible choices for the units digit: 0, 2, 4, 6, 8.
3. For each combination of thousands and units digits, ensure they are different.
4. Then, figure out the number of ways to choose the middle two digits (hundreds and tens places) from the remaining digits.

But maybe I should structure this more methodically. Let's consider each case for the thousands digit separately because each case might have different constraints, especially regarding the units digit.

**Case 1: Thousands digit is 4**

If the thousands digit is 4, then the units digit can be 0, 2, 6, or 8. Why? Because 4 is already used as the thousands digit, so the units digit can't be 4. So, out of the original 5 choices (0, 2, 4, 6, 8), we remove 4, leaving us with 4 choices.

Once we've chosen the thousands and units digits, we have to choose the hundreds and tens digits from the remaining digits. Since we've already used two digits (thousands and units), there are 8 remaining digits (since digits can range from 0-9, inclusive, but we've used two already). However, we have to make sure that the hundreds and tens digits are different from each other and from the thousands and units digits.

So, for the hundreds place, we can choose any digit except the thousands and units digits. That leaves us with 8 choices. Then, for the tens place, we can choose any digit except the thousands, units, and hundreds digits. That leaves us with 7 choices.

Wait, hold on. Is that correct? Let me think. If we've fixed the thousands and units digits, then for the hundreds digit, we have 8 choices (since two digits are already used). Then, for the tens digit, we have 7 choices (since three digits are already used). So, the number of possibilities for the middle two digits is 8 * 7.

But hold on, is that accurate? Let me verify. Let's say the thousands digit is 4 and the units digit is 0. Then, the hundreds and tens digits can be any of the remaining digits: 1,2,3,5,6,7,8,9. So, 8 digits. So, hundreds place: 8 choices. Then, tens place: 7 choices. So, yes, 8*7.

Therefore, for each choice of units digit, we have 8*7 possibilities for the middle two digits. Since in this case, the units digit has 4 choices, the total number of numbers in this case is 4 * 8 * 7.

Wait, no, actually, the 8*7 is for each specific combination of thousands and units digits. So, for each of the 4 units digits, we have 8*7 possibilities. So, total numbers in Case 1: 4 * 8 * 7.

Let me calculate that: 4 * 8 is 32, 32 * 7 is 224. Hmm, 224 numbers? That seems high. Wait, maybe I made a mistake.

Hold on, no. Wait, 8 choices for hundreds, 7 for tens, so 56 possibilities per units digit. And with 4 units digits, that's 4 * 56 = 224. Hmm, okay, that seems correct.

**Case 2: Thousands digit is 5**

Alright, moving on to thousands digit 5. Now, since 5 is an odd digit, it doesn't affect the parity of the number, but it does mean that the units digit has to be even, just like in the previous case.

So, units digit can be 0, 2, 4, 6, or 8. However, since the thousands digit is 5, which is different from all the units digit choices (0,2,4,6,8), we don't have to exclude any. So, that gives us 5 choices for the units digit.

Now, similar to before, after choosing the thousands and units digits, we have to choose the hundreds and tens digits from the remaining digits.

So, for the hundreds digit, we can choose any digit except 5 and the units digit. That leaves us with 8 choices. Then, for the tens digit, we can choose any digit except 5, the units digit, and the hundreds digit. So, 7 choices.

Therefore, for each units digit, we have 8 * 7 = 56 possibilities. With 5 units digits, that's 5 * 56 = 280 numbers.

Wait, 280? That seems even higher. Hmm, let me think again. If thousands digit is 5, units digit can be 0,2,4,6,8. So, 5 choices. Then, for each, hundreds digit: 8 choices, tens digit: 7. So, 5*8*7=280. That seems correct.

But hold on, does choosing 0 as the units digit affect anything else? For example, if units digit is 0, is there any restriction on the hundreds or tens digits? No, because 0 is allowed in the hundreds and tens places, as long as they aren't the same as the thousands or units digits.

So, yes, 5 units digits, each with 56 possibilities, so 280.

**Case 3: Thousands digit is 6**

Now, moving on to thousands digit 6. Similar to the first case, since 6 is even, the units digit can't be 6 because all digits must be different. So, the units digit can be 0, 2, 4, or 8. So, 4 choices.

Then, similar to Case 1, after choosing thousands and units digits, we have 8 choices for the hundreds digit and 7 for the tens digit.

Therefore, for each units digit, 8*7=56 possibilities. With 4 units digits, that's 4*56=224 numbers.

Wait, same as Case 1. Makes sense because 6 is similar to 4 in terms of being even, so same number of units digits.

So, summarizing:

- Case 1 (4xxx): 4 units digits, each with 56 numbers: 224
- Case 2 (5xxx): 5 units digits, each with 56 numbers: 280
- Case 3 (6xxx): 4 units digits, each with 56 numbers: 224

So, total numbers = 224 + 280 + 224. Let's add those up.

224 + 224 = 448, then 448 + 280 = 728. So, 728 numbers?

Wait, that seems a bit high. Let me see if I considered all constraints correctly.

Wait, hold on. When the thousands digit is 4 or 6, we have 4 choices for the units digit. But is that always the case?

Wait, when thousands digit is 4, units digits can be 0,2,6,8. Similarly, when thousands digit is 6, units digits can be 0,2,4,8. So, that's 4 choices each, correct.

But wait, hold on. When thousands digit is 4, can units digit be 6? Yes, because 6 hasn't been used yet. Similarly, when thousands digit is 6, units digit can be 4. So, that's fine.

But wait, is 0 allowed as a units digit? Yes, because the number is still four digits, and 0 in the units place is fine. The only restriction is that all digits must be unique.

So, perhaps my initial calculation is correct.

But let me think about the total number of four-digit numbers between 4000 and 7000. The range is 4000 to 6999, inclusive. So, total numbers: 6999 - 4000 + 1 = 3000 numbers.

Out of these, how many are even? Well, every other number is even, so approximately 1500. But we're further restricting that all digits are different.

So, 728 is less than 1500, which is reasonable. So, maybe 728 is correct.

But let me think again about the calculations for each case.

**Case 1: Thousands digit is 4**

- Units digit can be 0,2,6,8: 4 choices
- Hundreds digit: 8 choices (since two digits are already used: 4 and the units digit)
- Tens digit: 7 choices (since three digits are already used)
- So, total: 4 * 8 * 7 = 224

**Case 2: Thousands digit is 5**

- Units digit can be 0,2,4,6,8: 5 choices
- Hundreds digit: 8 choices (digits used: 5 and units digit)
- Tens digit: 7 choices (digits used: 5, units, and hundreds)
- Total: 5 * 8 * 7 = 280

**Case 3: Thousands digit is 6**

- Units digit can be 0,2,4,8: 4 choices (excluding 6)
- Hundreds digit: 8 choices
- Tens digit: 7 choices
- Total: 4 * 8 * 7 = 224

So, adding all together: 224 + 280 + 224 = 728

Wait, but hold on a second. Is there a possibility that 0 is being counted as a digit in the thousands place? No, because thousands digit is fixed as 4,5,6. So, 0 can only be in the hundreds, tens, or units place.

But in the hundreds or tens place, 0 is allowed. So, my earlier calculation should be okay because when I choose the hundreds digit, I include 0 as a possible choice, as long as it's not already used in units or thousands place.

Wait, but in the hundreds place, 0 is allowed as long as it's not already used. Similarly for the tens place.

So, my calculation of 8 choices for hundreds and 7 for tens is correct because in each case, two digits are already used (thousands and units), so 10 - 2 = 8 remaining digits for hundreds, then 7 for tens.

So, 8*7 for each combination of thousands and units digits.

Therefore, 224 + 280 + 224 = 728.

Wait, but I just realized something. When the units digit is 0, is 0 allowed in the hundreds or tens place? Yes, because 0 is only used in the units place, so the hundreds and tens digits can still be 0 if they haven't been used yet. Wait, no, because 0 is already used in the units place, so it can't be used again in hundreds or tens. So, actually, when units digit is 0, the hundreds and tens digits can't be 0. So, does that affect the number of choices?

Wait, in my initial calculation, I considered that for hundreds digit, there are 8 choices regardless of what the units digit is. But actually, if the units digit is 0, then hundreds digit can be any digit except 4 (or 5,6 depending on thousands digit) and 0. So, that would be 8 choices, same as before.

Wait, hold on. Let's take an example.

If thousands digit is 4 and units digit is 0, then hundreds digit can be any digit except 4 and 0: so 1,2,3,5,6,7,8,9: 8 digits. Similarly, if units digit is 2, hundreds digit can be any digit except 4 and 2: 8 digits. So, yes, regardless of what the units digit is, hundreds digit has 8 choices because two digits are already used (thousands and units), so 10 - 2 = 8.

Similarly, for tens digit, it's 7 choices, since three digits are already used.

Therefore, my initial calculation is correct: 8*7 for each case.

So, 728 seems to be the total number.

But wait, just to ensure, let me think about another way.

Total four-digit numbers between 4000 and 7000: 3000 numbers.

Number of even numbers: 1500. But we need the numbers where all digits are distinct.

Alternatively, maybe I can calculate the number of such numbers by considering the restrictions.

But perhaps another approach is to calculate the number of even numbers with distinct digits between 4000 and 7000.

So, the number is even, so last digit must be 0,2,4,6,8.

First digit: 4,5,6.

So, the number of such numbers is equal to:

Sum over first digit (4,5,6) of [number of choices for units digit * number of permutations for the middle two digits]

Which is exactly what I did earlier.

So, for first digit 4:

Units digit: 0,2,6,8 (4 choices)

Middle digits: 8 * 7 = 56

So, 4 * 56 = 224

Similarly, for first digit 5:

Units digit: 0,2,4,6,8 (5 choices)

Middle digits: 8 * 7 = 56

So, 5 * 56 = 280

For first digit 6:

Units digit: 0,2,4,8 (4 choices)

Middle digits: 8 * 7 = 56

So, 4 * 56 = 224

So, total: 224 + 280 + 224 = 728

Therefore, 728 numbers.

Wait, but let me think: is there any case where the hundreds or tens digit can't be 0? For example, in the hundreds place, 0 is allowed, right? Because 4000 is a valid number, but in that case, digits are not unique. Wait, in our calculation, we've already considered that all digits must be unique, so 0 is only allowed once.

Wait, for example, in the case of thousands digit 4, units digit 0: then hundreds and tens digits can't be 4 or 0, so they can be 1,2,3,5,6,7,8,9: 8 digits, which is correct.

Similarly, if units digit is 2, then hundreds and tens can't be 4 or 2, so 8 digits: 0,1,3,5,6,7,8,9: 8 digits.

Wait, but if units digit is 0, then 0 is already used, so hundreds and tens can't be 0 or 4. But in that case, hundreds digit can be 1,2,3,5,6,7,8,9: 8 digits, same as before. So, regardless of whether units digit is 0 or not, we have 8 choices for hundreds and 7 for tens.

Therefore, the total is 728.

But let me test this with a smaller case to see if my reasoning is correct.

Suppose I want to count the number of even numbers between 4000 and 5000 with all distinct digits.

Thousands digit: 4

Units digit: 0,2,6,8 (4 choices)

For each units digit, hundreds and tens digits can be any of the remaining 8 digits.

Wait, 8*7=56, so total numbers: 4*56=224, which is consistent with my earlier calculation.

Similarly, for numbers between 5000 and 6000, thousands digit is 5, units digit: 0,2,4,6,8 (5 choices), each with 56, so 280.

Between 6000 and 7000, thousands digit is 6, units digit: 0,2,4,8 (4 choices), each with 56, so 224.

Thus, 224 + 280 + 224 = 728.

Wait, but wait another thing: when the units digit is 0, does that affect the hundreds or tens digits? For example, if thousands digit is 4 and units digit is 0, then hundreds and tens digits can be any of 1,2,3,5,6,7,8,9: 8 digits. So, 8*7=56.

Similarly, if units digit is 2, then hundreds and tens digits can be 0,1,3,5,6,7,8,9: 8 digits. So, same.

So, seems okay.

Wait, but another thought: when units digit is 0, hundreds digit can be 0? No, because units digit is 0, so hundreds digit can't be 0. Wait, no, in the case where units digit is 0, hundreds digit can be 0? Wait, no, because digits must be unique.

Wait, no. If units digit is 0, then hundreds digit can't be 0, because 0 is already used. Similarly, if units digit is 2, hundreds digit can be 0, because 2 is already used, but 0 is still available.

Wait, so in the case where units digit is 0, hundreds digit can't be 0, so hundreds digit has 8 choices (excluding thousands and units digits: 4 and 0). Similarly, if units digit is non-zero, hundreds digit can be 0, but must exclude thousands and units digits.

Wait, but in both cases, whether units digit is 0 or not, the number of available digits for hundreds and tens is 8 and 7.

Because, in the case of units digit being 0: hundreds digit can be any digit except 4 and 0: 8 choices. Then, tens digit can be any digit except 4, 0, and the hundreds digit: 7 choices.

In the case of units digit being non-zero (say 2): hundreds digit can be any digit except 4 and 2: 8 choices (including 0). Then, tens digit can be any digit except 4, 2, and the hundreds digit: 7 choices.

So, regardless of whether units digit is 0 or not, the number of possibilities is 8*7.

Therefore, my initial calculation is correct.

Thus, the total number is 728.

Wait, but hold on. Let me check with a specific example. Let's say thousands digit is 4 and units digit is 0. Then, hundreds and tens digits can be any of 1,2,3,5,6,7,8,9: 8 digits for hundreds, then 7 for tens. So, 56 numbers.

Similarly, if thousands digit is 4 and units digit is 2: hundreds digit can be 0,1,3,5,6,7,8,9: 8 digits, then 7 for tens.

Same for units digits 6 and 8.

Similarly, for thousands digit 5, units digit 0,2,4,6,8: each case, hundreds and tens digits have 8 and 7 choices.

Same for thousands digit 6.

Therefore, the calculation seems consistent.

Therefore, my conclusion is that there are 728 such numbers.

But wait, hold on, I just thought of something. When the thousands digit is 4 or 6, and the units digit is 0, does that affect the hundreds or tens digits? For example, if units digit is 0, then 0 is already used, so hundreds and tens digits can't be 0. But since 0 is already excluded in the hundreds digit selection (since we're choosing from 8 digits excluding thousands and units digits), which already excludes 0 if units digit is 0.

Wait, no, when units digit is 0, thousands digit is 4 or 6, so hundreds digit can be 1,2,3,5,6,7,8,9: 8 digits, which includes 0? Wait, no, 0 is already used in units digit, so hundreds digit can't be 0 or thousands digit.

So, 8 digits, which are 1,2,3,5,6,7,8,9 if units digit is 0, or 0,1,3,5,6,7,8,9 if units digit is non-zero.

Wait, but in both cases, the number of choices is 8 for hundreds digit, because whether 0 is allowed or not, we have 8 digits to choose from.

Wait, for example:

Case 1: thousands=4, units=0. Hundreds can be 1,2,3,5,6,7,8,9: 8 choices.

Case 2: thousands=4, units=2. Hundreds can be 0,1,3,5,6,7,8,9: 8 choices.

So, regardless, 8 choices for hundreds, 7 for tens.

Therefore, my calculation is correct.

Therefore, the total number is 728.

But just to be thorough, let me think about the total number of four-digit numbers between 4000 and 7000 with all distinct digits and even.

Another way to compute this is:

Total number of even four-digit numbers between 4000 and 7000 with distinct digits.

Number of choices for thousands digit: 3 (4,5,6)

Number of choices for units digit: depends on thousands digit.

If thousands digit is 4 or 6: units digit can be 0,2,4,6,8 excluding the thousands digit: 4 choices.

If thousands digit is 5: units digit can be 0,2,4,6,8: 5 choices.

Then, for the middle two digits: 8 choices for hundreds, 7 for tens.

So, total numbers:

For thousands digits 4 and 6: 2 digits, each with 4 units digits, each with 8*7=56: 2*4*56=448

For thousands digit 5: 1 digit, with 5 units digits, each with 56: 1*5*56=280

Total: 448 + 280=728

Same as before.

Therefore, I can be confident that 728 is the correct answer.

Wait, hold on, just another sanity check: 728 is less than 3000, which is the total number of four-digit numbers between 4000 and 7000. So, 728 is about a quarter of that, which seems plausible.

Given that about half of the numbers are even, and then among those, a certain number have all distinct digits.

Therefore, 728 seems correct.

So, after careful consideration and multiple ways of calculating, I think the answer is 728.

**Final Answer**
The number of such even integers is \boxed{728}.
</think>

To determine how many even integers between 4000 and 7000 have four different digits, we consider the following cases based on the thousands digit (4, 5, or 6) and the units digit (0, 2, 4, 6, or 8).

**Case 1: Thousands digit is 4**

- Units digit can be 0, 2, 6, or 8 (4 choices).
- For each choice of units digit, the hundreds digit has 8 choices (excluding the thousands and units digits), and the tens digit has 7 choices (excluding the thousands, units, and hundreds digits).
- Total for this case: \(4 \times 8 \times 7 = 224\).

**Case 2: Thousands digit is 5**

- Units digit can be 0, 2, 4, 6, or 8 (5 choices).
- For each choice of units digit, the hundreds digit has 8 choices, and the tens digit has 7 choices.
- Total for this case: \(5 \times 8 \times 7 = 280\).

**Case 3: Thousands digit is 6**

- Units digit can be 0, 2, 4, or 8 (4 choices).
- For each choice of units digit, the hundreds digit has 8 choices, and the tens digit has 7 choices.
- Total for this case: \(4 \times 8 \times 7 = 224\).

Summing the totals from all cases:

- Total number of such even integers: \(224 + 280 + 224 = 728\).

Thus, the number of even integers between 4000 and 7000 with four different digits is \(\boxed{728}\)."""

input_ids = text_to_input_ids(text)

import transformer_utils.logit_lens
import importlib

importlib.reload(transformer_utils.logit_lens)
from transformer_utils.logit_lens import plot_logit_lens

# for start in range(0, 5800, 25):
#     plot_logit_lens(model, tokenizer, input_ids, start_ix=start, end_ix=start + 25, probs=True)

print(path)
# Mid-text, with occurrence of answer (728)
start = 5265
plot_logit_lens(
    model, tokenizer, input_ids, start_ix=start, end_ix=start + 25, probs=True
)

# End of text
start = 5805
plot_logit_lens(
    model, tokenizer, input_ids, start_ix=start, end_ix=start + 23, probs=True
)
# %%
path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(path)
# path = "Qwen/Qwen2.5-Math-1.5B"
model = transformers.AutoModelForCausalLM.from_pretrained(path).cuda()

# Mid-text, with occurrence of answer (728)
start = 5265
plot_logit_lens(
    model, tokenizer, input_ids, start_ix=start, end_ix=start + 25, probs=True
)

# End of text
start = 5805
plot_logit_lens(
    model, tokenizer, input_ids, start_ix=start, end_ix=start + 23, probs=True
)
# %%
