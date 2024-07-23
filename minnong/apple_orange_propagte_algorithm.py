from minnong.Propagation import *

apple_num = 2
orange_num = 3
apple = 100
orange = 150
tax = 1.1 # 10% tax

# 연산자 지정
mul_apple_layer = Mullayer()
mul_orange_layer = Mullayer()
add_apple_orange_layer = Addlayer()
mul_tax_layer = Mullayer()

# forward_pass
apple_price = mul_apple_layer.forward(apple,apple_num)
orange_price = mul_orange_layer.forward(orange,orange_num)
sum_price = add_apple_orange_layer.forward(apple_price,orange_price)
total_price = mul_tax_layer.forward(sum_price,tax)

print(total_price)

# backward_pass
# 순서가 헷갈리면 안된다.
dtotal_price = 1
dsum_price , dtax  = mul_tax_layer.backward(dtotal_price)
dapple_price , dorange_price = add_apple_orange_layer.backward(dsum_price)
dapple , dapple_num = mul_apple_layer.backward(dapple_price)
dorange , dorange_num = mul_orange_layer.backward(dorange_price)

print(dapple, dorange, dapple_num, dorange_num ,dtax)
