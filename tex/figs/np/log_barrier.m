u=-3:0.001:0
figure(1);clf;
hold on
for lambda = 1:3
plot(u,-lambda*log(-u),'k','linewidth',1)
end
plot([-3,0],[0,0],'--k')
plot([0,0],[0,10],'--k')
hold off
xlabel('u')
title('-\lambda log(-u)')
text(0,25,'\infty')
set(gcf,'color','w')
axis([-3 1 -5 10])