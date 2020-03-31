#ifndef MM_LANG_H
#define MM_LANG_H

namespace mmatrix {
namespace lang {

/**
 * sparse_matrix<3,2,1> A;
 * matrix<4,1> B;
 * function F(matrix<4,3,2> X) =
 *   X *{2} A + B;
 *
 * function G(matrix<4,3,2> X) =
 *   derivative F by X; 
 *
 * Note that the derivative expression is scoped to inner F.
 * 
 * EXPR = [VAR_DEC | FUNCTION]
 * VAR_DEC = TYPE VAR_NAME ';'
 * TYPE = TYPE_KW (SHAPE)
 * TYPE_KW = ['sparse_matrix' | 'matrix']
 * SHAPE = '<' INT_LIST '>'
 * INT_LIST = INT (',' INT_LIST)
 * FUNCTION = 'function' FUNC_NAME '(' PARAM_LIST ')' '=' FUNC_EXPR ';'
 * FUNC_EXPR = ('(') [SUM | SUB | PROD | DERIVATIVE] (')')
 **/

}  // namespace lang
}  // namespace mmatrix

#endif
