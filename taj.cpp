#include<bits/stdc++.h>
# define ll long long 
using namespace std;




void test(){
    
    ll x,y;
    cin>>x>>y;

    ll m=max(x,y);

    ll q=m*m-(m-1);

    if(x==y)
    {
        cout<<q<<endl;
        return ;
    }

    bool l=false;
    bool u=false;

    bool even=false;

    if(m%2==0)
    {
        even=true;
    }

    if(m>x){
        u=true;
    }
    if(m>y)
    {
        l=true;
    }

    // cout<<q<<u<<l<<endl;
    if(even)
    {
        if(u){
            cout<<q-(m-x)<<endl;
        }
        else{
            cout<<q+(m-y)<<endl;
        }
    }
    else{
            if(u){
            cout<<q+(m-x)<<endl;
        }
        else{
            cout<<q-(m-y)<<endl;
        }
    }

return ;

}
int main()
{
   int t;
   cin>>t;
   while(t--)
   {
        test();
   }
   return 0;

}

