def build_conv(env, query):
    if env == "R1":
        dialogue = [
            {
                "role": "user",
                "content": query,
            }
        ]
    elif env == "MATH-0-shot":
        dialogue = [
            {
                "role": "user",
                "content": """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$.

Where [answer] is just the final number or expression that solves the problem.

Problem: """
                + query,
            }
        ]
    elif env == "MATH-1-shot":
        dialogue = [
            {
                "role": "user",
                "content": """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$.

Where [answer] is just the final number or expression that solves the problem.

Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.""",
            },
            {
                "role": "assistant",
                "content": """## Step 1: Consider the expression inside the first square root
The expression inside the first square root is $x-2$, and it must be non-negative. This means that $x-2 \\ge 0$, which simplifies to $x \\ge 2$.

## Step 2: Consider the expression inside the second square root
The expression inside the second square root is $5-x$, and it must be non-negative. This means that $5-x \\ge 0$, which simplifies to $x \\le 5$.

## Step 3: Consider the denominator of the expression
The denominator of the expression is $\\sqrt{5-x}$, and it cannot be equal to zero. This means that $5-x>0$, which simplifies to $x<5$.

## Step 4: Combine the results from the previous steps
Combining the results from the previous steps, we have $x \\ge 2$ and $x < 5$. This means that the domain of the expression is $[2,5)$.

The final answer is: $\\boxed{[2,5)}$.""",
            },
            {
                "role": "user",
                "content": """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$.

Where [answer] is just the final number or expression that solves the problem.

Problem: """
                + query,
            },
        ]
    elif env == "MATH-4-shot":
        dialogue = [
            {
                "role": "user",
                "content": """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$.

Where [answer] is just the final number or expression that solves the problem.

Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.""",
            },
            {
                "role": "assistant",
                "content": """## Step 1: Consider the expression inside the first square root
The expression inside the first square root is $x-2$, and it must be non-negative. This means that $x-2 \\ge 0$, which simplifies to $x \\ge 2$.

## Step 2: Consider the expression inside the second square root
The expression inside the second square root is $5-x$, and it must be non-negative. This means that $5-x \\ge 0$, which simplifies to $x \\le 5$.

## Step 3: Consider the denominator of the expression
The denominator of the expression is $\\sqrt{5-x}$, and it cannot be equal to zero. This means that $5-x>0$, which simplifies to $x<5$.

## Step 4: Combine the results from the previous steps
Combining the results from the previous steps, we have $x \\ge 2$ and $x < 5$. This means that the domain of the expression is $[2,5)$.

The final answer is: $\\boxed{[2,5)}$.""",
            },
            {
                "role": "user",
                "content": """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$.

Where [answer] is just the final number or expression that solves the problem.

Problem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).""",
            },
            {
                "role": "assistant",
                "content": """## Step 1: Understand the properties of determinants
We know that the determinant of a product of two matrices is equal to the product of their determinants.

## Step 2: Apply the property to the given matrices
Using the property, we can write $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B})$.

## Step 3: Substitute the given values
We are given that $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12$. Substituting these values, we get $\\det (\\mathbf{A} \\mathbf{B}) = (2)(12)$.

## Step 4: Calculate the final answer
Evaluating the product, we find that $\\det (\\mathbf{A} \\mathbf{B}) = 24$.

The final answer is: $\\boxed{24}$.""",
            },
            {
                "role": "user",
                "content": """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$.

Where [answer] is just the final number or expression that solves the problem.

Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?""",
            },
            {
                "role": "assistant",
                "content": """## Step 1: Calculate the total weight Terrell lifts with two 20-pound weights
Terrell lifts two 20-pound weights 12 times, so the total weight he lifts is $2 \\cdot 12 \\cdot 20 = 480$ pounds.

## Step 2: Calculate the total weight Terrell lifts with two 15-pound weights for n times
If Terrell lifts two 15-pound weights instead for $n$ times, he will lift a total of $2 \\cdot 15 \\cdot n = 30n$ pounds of weight.

## Step 3: Equate the total weight lifted with 15-pound weights to 480 pounds and solve for n
Equating the total weight lifted with 15-pound weights to 480 pounds, we can solve for $n$: $30n = 480$.

## Step 4: Solve for n
To solve for $n$, divide both sides by 30: $n = \\frac{480}{30}$.

## Step 5: Calculate the value of n
The value of $n$ is $\\frac{480}{30} = 16$.

The final answer is: $\\boxed{16}$.""",
            },
            {
                "role": "user",
                "content": """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$.

Where [answer] is just the final number or expression that solves the problem.

Problem: If the system of equations

\\begin{align*}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.""",
            },
            {
                "role": "assistant",
                "content": """## Step 1: Understand the problem
We are given a system of linear equations with two variables, $x$ and $y$, and two constants, $a$ and $b$. Our goal is to find the ratio $\\frac{a}{b}$, assuming $b$ is nonzero.

## Step 2: Multiply the first equation by a suitable factor
We multiply the first equation by $-\\frac{3}{2}$ to obtain an equation with the same coefficients for $y$ as the second equation: $$-\\frac{3}{2}(6x-4y)=a\\Rightarrow 6y-9x=-\\frac{3}{2}a.$$

## Step 3: Equate the right-hand sides of the two equations
Since we know that $6y-9x=b$ from the second equation, we can equate the right-hand sides of the two equations to obtain $$-\\frac{3}{2}a=b.$$

## Step 4: Solve for the ratio $\\frac{a}{b}$
Finally, we can solve for the ratio $\\frac{a}{b}$ by dividing both sides of the equation by $b$, giving us $$\\frac{a}{b}=-\\frac{2}{3}.$$

The final answer is: $\\boxed{-\\frac{2}{3}}.""",
            },
            {
                "role": "user",
                "content": """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$.

Where [answer] is just the final number or expression that solves the problem.

Problem: """
                + query,
            },
        ]
    elif env == "backtrack":
        dialogue = [
            {
                "role": "user",
                "content": """You are an AI language model designed to assist with math problem-solving. In this task, I will provide you with math problems. Your goal is to solve the problem step-by-step, optionally backtracking to previous states if the current steps are unlikely to lead to a correct solution. Each step should be labeled sequentially as Node #. After you have finished solving the problem, present your final answer as \boxed{Your Answer}.

Here is the math word problem for you to solve:
"""
                + query,
            }
        ]
    return dialogue
