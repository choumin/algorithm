#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_EXPR_LEN	256
#define MAX_ELEM_COUNT	-128

#define TRUE	1
#define FALSE	0

#define NO_ERR		-1
#define MA_MAROOM	0
#define MA_STHEAD	1
#define MA_STDATA	2

typedef union ELEMTYPE{
	char operator;
	double operand;
}ELEMTYPE;

typedef struct stack{
	ELEMTYPE *data;
	int top;
	int max_room;
}STACK;

int errno = NO_ERR;

const char *errmsg[] = {
	"malloc stack max room error!"
	"malloc stack head error!",
	"malloc stack data error!",
	
};
int init_stack(STACK **stack, int max_room)
{	
	if (max_room < 0 || max_room > MAX_ELEM_COUNT) {
		errno = MA_MAROOM;
		return FALSE;
	}
	*stack = (STACK *)malloc(sizeof(STACK));
	if (*stack == NULL) {
		errno = MA_STHEAD;
		return FALSE;
	} 
	(*stack)->data = (ELEMTYPE *)malloc(sizeof(ELEMTYPE) * max_room);
	if ((*stack)->data == NULL) {
		errno = MA_STDATA;
		free(*stack);	
		return FALSE;
	}
	(*stack)->max_room = max_room;
	(*stack)->top = 0;
	
	return TRUE;
}
void destory_stack(STACK **stack)
{
	if (stack) {
		if (*stack) {
			if ((*stack)->data)
				free((*stack)->data);
			free(*stack);
			(*stack) = NULL;
		}
	}
}
int is_stack_full(STACK s)
{
		
	return s.max_room <= s.top;
}
int is_stack_empty(STACK s)
{
	return s.top == 0;
}
int push(STACK *s, ELEMTYPE va)
{
	int OK = TRUE;

	if (s && !is_stack_full(*s))
		s->data[s->top++] = va;
	else
		OK = FALSE;
	return OK;
}
int pop(STACK *s, ELEMTYPE *va)
{
	int OK = TRUE;

	if (s && !is_stack_empty(*s))
		(*va) = s->data[--s->top];
	else 
		OK = FALSE;

	return OK;
}
int readtop(STACK *s, ELEMTYPE *va)
{
	int OK = TRUE;

	if (s && !is_stack_empty(*s))
		(*va) = s->data[s->top - 1];
	else 
		OK = FALSE;
	return OK;
}

int check_expr(char *expr)
{
	int i = 0;
	int bracket_count = 0;

	while (expr[i]) {
		if (bracket_count < 0) {
	
			printf("bracket don't match!\n");
			return -1;
		}
		if (expr[i] == '(')
			bracket_count++;
		else if (expr[i] == ')')
			bracket_count--;
		i++;
	}
	return 0;
}
void SUCCESS_EXIT(STACK **num_stack, STACK **oper_stack)
{

	destory_stack(num_stack);
	destory_stack(oper_stack);
}
void ERR_EXIT(STACK **num_stack, STACK **oper_stack)
{
	printf("errno = %d: %s\n", errno, errmsg[errno]);
	
	destory_stack(num_stack);
	destory_stack(oper_stack);
	exit(-1);
}
int process_expression(char *expr_str, STACK *oper_stack, STACK *num_stack, double *ret)
{
	int i = 0;
	int OK = TRUE;
 	int state = BEGIN;
	ELEMTYPE value = {0,0};
	int bracket_count = 0;
	int FINISHED = FALSE;

	while (!FINISHED && OK) {
		i = skip_blank(expr_str, i);
		if (state == BEGIN) {
			if (expr_str[i] == '(') {
				bracket_count++;
				value.op = '(';
				push(oper_stack, value);
			} else if (expr_str[i] == '+' || expr_str[i] == '-') {
				
			} else if (expr_str[i] == ')') {
			
			}
				
		} else if (state == AFTER_NUM) {

		} else if (state == END) {
		} 
		if (bracket_count < 0) 
			OK = FALSE;
	}
	return OK;
}
int main(void)
{
	char expression[MAX_EXPR_LEN] = {0};	
	int OK = TRUE;
	STACK *operator_stack = NULL;
	STACK *operand_stack = NULL;
	double ret = 0;	

	fgets(expression, MAX_EXPR_LEN, stdin);
	OK = init_stack(&operator_stack, MAX_ELEM_COUNT);
	if (!OK) 
		ERR_EXIT(&operand_stack, &operator_stack);

	OK = init_stack(&operand_stack, MAX_ELEM_COUNT);
	if (!OK) 
		ERR_EXIT(&operand_stack, &operator_stack);
	OK = process_expression(expression, operator_stack, operand_stack, &ret);
	if (!OK)
		ERR_EXIT(&operand_stack, &operator_stack);
	else 
		printf("%s = %lf\n", expression, ret);
	SUCCESS_EXIT(&operand_stack, &operator_stack);	
	return 0;
}
