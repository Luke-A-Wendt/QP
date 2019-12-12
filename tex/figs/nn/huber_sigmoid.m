t=2;
u=-10*t:0.1:10*t;
y=[];
for i=1:length(u)
    if abs(u(i))<t
        y=[y,u(i)/2];
    else
        y=[y,t*sign(u(i))-t^2/u(i)/2];
    end
end
plot(u,y,'k','linewidth',2)
hold on
% plot([-t,t],[t^2,t^2],'--k')
% plot([t,t],[0,t^2],'--k')
% plot([-t,-t],[0,t^2],'--k')
plot(1,1/2,'ok','markerfacecolor','w','linewidth',2)
plot(-1,-1/2,'ok','markerfacecolor','w','linewidth',2)
hold off
xticks([0])
yticks([-1/2,0,1/2])
text(-t-0.5,-t-.15,'-t')
text(t,-t-.15,'t')
text(min(u)-4.5,-t,'-t')
text(min(u)-4.5,t,'t')
% text(-2-.3,t^2,'t^2')
xlabel('u')
ylabel('y')
title('f(u)/2u')
set(gcf,'color','w')