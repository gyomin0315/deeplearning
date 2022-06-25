%%t = 1:0.01:2;

t2 = 0:0.1:2*pi;

%%x = sin(2*pi*t);

x2 = sin(2*t2);


figure
plot(t2,x2)
%%w_noise = wgn(1,101,-20);

w_noise2 = wgn(1,length(t2),-20)*10;

w_noise = w_noise*10;


hold on
% % % % plot(t,(x+w_noise))
% % % % legend('Sine Wave','Sine Wave with Noise')

y2 = x2+w_noise2;