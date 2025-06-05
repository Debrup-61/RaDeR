prompt_summarize = '''Given the statement of a mathematical theorem in a structured latex format, convert it to a simpler natural language format by removing latex notations.

Theorem: \section{Quadratic Irrational is Root of Quadratic Equation} Tags: Algebra, Quadratic Equations, Quadratic Irrationals \\begin{theorem} Let $x$ be a quadratic irrational. Then $x$ is a solution to a quadratic equation with rational coefficients. \end{theorem}
Natural language theorem: Quadratic Irrational is Root of Quadratic Equation - A quadratic irrational number is always the root of some quadratic equation with rational coefficients.

Theorem: \section{Null Ring is Commutative Ring} Tags: Null Ring, Commutative Rings \\begin{theorem} Let $R$ be the null ring. That is, let: :$R := \struct {\set {0_R}, +, \circ}$ where ring addition and ring product are defined as: :$0_R + 0_R = 0_R$ :$0_R \circ 0_R = 0_R$ Then $R$ is a commutative ring. \end{theorem}
Natural language theorem: Null Ring is Commutative Ring - The null ring, where the only element is 0 and addition and multiplication are defined as 0 + 0 = 0 and 0 * 0 = 0, is a commutative ring.

Theorem: \section{Square Inscribed in Circle is greater than Half Area of Circle} Tags: Circles, Squares \\begin{theorem} A square inscribed in a circle has an area greater than half that of the circle. \end{theorem}
Natural language theorem: Square Inscribed in Circle is greater than Half Area of Circle -  A square inscribed in a circle has an area greater than half the area of the circle.'''