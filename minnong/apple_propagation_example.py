from minnong.Propagation import *

apple = 100
apple_num = 2
tax = 1.1

# forward propagatuion -> 순서대로 연산해야한다.
mul_apple_layer = Mullayer()
mul_tax_layer = Mullayer()
# 아직은 넣어준 값이 없어서 아무런 출력이 없다.

apple_price = mul_apple_layer.forward(apple,apple_num)
# 클래스 내 forward 함수에는 x*y 기능이 들어있다.
price =  mul_tax_layer.forward(apple_price,tax)
# 세금과 방금 뒤에서 계산한 값을 받아서 함수에 넣고 , 최종값을 계산해준다.
print(int(price))

# Backpropagation
dprice = 1
dapple_price , dtax = mul_tax_layer.backward(dprice)
# global gradinet를 인수로써 받는다. local gradient는 이미 저장되어있으므로 굳이 저장하지 않는다
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# 여기서 쓰이는 backward 함수 내에는 dx = dapple_price * y , dy = dapple_price_x 로 만들어주는 식이 저장되어있다.
# 아까 x,y를 저장해놓은 상태이기 때문에 굳이 x,y를 따로 넣을 필요는 없다.

print(dapple, dapple_num , dtax)

