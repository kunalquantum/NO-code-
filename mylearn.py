import streamlit as st

#--page setup 
about_page=st.Page(
    #linking of the page 
    page="prod/aboutMe.py",
    title="About Me",
    icon=":material/thumb_up:",
   

)


data=st.Page(
    #linking of the page 
    page="prod/dataF.py",
    title="Data",
    icon=":material/thumb_up:",
    default=True
)
prep=st.Page(
    #linking of the page 
    page="prod/prep1.py",
    title="preprocessing",
    icon=":material/thumb_up:",

)
feature=st.Page(
    #linking of the page 
    page="prod/feature.py",
    title="Feature Selection",
    icon=":material/thumb_up:",

)

modal=st.Page(
    #linking of the page 
    page="prod/modal.py",
    title="Modal Selection",
    icon=":material/thumb_up:",

)

compare=st.Page(
    #linking of the page 
    page="prod/compareModals.py",
    title="Modal Comparison",
    icon=":material/thumb_up:",



)
#smart data hub
test=st.Page(
    #linking of the page 
    page="prod/test.py",
    title="Train / Test",
    icon=":material/thumb_up:",



)



# navigation bar 

pg=st.navigation(
   {
       "Info":[about_page],
       "No Code Ml":[data,prep,feature,modal,compare,test]
   }
    )

st.image("prod/assets/logo3.jpg")
st.sidebar.text("Made with ❤ by Kunal")
logo_url="prod/assets/logo3.jpg"


import streamlit as st

# Your main app code here

# Footer message
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        margin-right:1550px;
        width: 100%;
        text-align: center;
        font-size: large;
        color: gray;
    }
    </style>
    <div class="footer">
        © 2024 Kunal Sharma. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)

pg.run()