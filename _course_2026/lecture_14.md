---
lecture_num: '14'
title: 'SVD & PCA: Dimensionality Reduction'
date: '2026-03-05T08:00:00+03:00'
type: lecture
tldr: Eigenvalues, singular value decomposition, principal component analysis, and
  low-rank approximation.
preparation:
- name: Steve Brunton - SVD YouTube series
  type: video
  label: recommended
  url: https://www.youtube.com/playlist?list=PLMrJAkhIeNNSVjnsviglFoY2nXildDCcv
material:
- name: Lecture Slides
  type: slides
  url: https://learnslides.onrender.com/slides/09_svd_pca
- name: PCA Reference (PDF)
  type: notes
  label: recommended
  url: /static_files/lectures/14/pca.pdf
- name: Lecture Recording — PCA (March 5)
  type: video
  url: https://teams.microsoft.com/l/meetingrecap?driveId=b%21bJ0d-pNdx0Ks8vhDBeIsafFHjccgzdhPua4HL33q-SRmGRJ8_wZ6SoitW7-b_rZt&driveItemId=01COK4GJFRBYG6WVUEJZHJRUEEVAXVW7MR&sitePath=https%3A%2F%2Fmailaub-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fjb50_aub_edu_lb%2FIQCxDg3rVoROTpjQhKgvW32RAbHCnpNI0KnOcEjBXG-saOA&fileUrl=https%3A%2F%2Fmailaub-my.sharepoint.com%2Fpersonal%2Fjb50_aub_edu_lb%2FDocuments%2FRecordings%2FData+Driven+Modeling+Lecture-20260305_141455-Meeting+Recording.mp4%3Fweb%3D1&iCalUid=040000008200E00074C5B7101A82E0080000000003EBBA7374ACDC01000000000000000010000000C4B8A8B50FDA704089EB494AE30B878F&threadId=19%3Ameeting_NWMyZWUwMzUtOWNlMS00MGQyLWFhMTMtZWMyMGMwZDE2YzA5%40thread.v2&organizerId=879bbc5b-4fb7-4d7a-9527-40ff9edea65e&tenantId=c7ba5b1a-41b6-43e9-a120-6ff654ada137&callId=05546393-3bc1-4fe8-afa4-609dccda4f44&threadType=meeting&meetingType=Scheduled&subType=RecapSharingLink_RecapCore
- name: Lecture Recording — SVD (March 10)
  type: video
  url: https://teams.microsoft.com/l/meetingrecap?driveId=b%21bJ0d-pNdx0Ks8vhDBeIsafFHjccgzdhPua4HL33q-SRmGRJ8_wZ6SoitW7-b_rZt&driveItemId=01COK4GJGK4DHN7NO7MJFIRIVW3UKE42P4&sitePath=https%3A%2F%2Fmailaub-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fjb50_aub_edu_lb%2FIQDK4M7ftd9iSoiitt0UTmn8AY0CPA4T9l_j_5D6PWCIjHs&fileUrl=https%3A%2F%2Fmailaub-my.sharepoint.com%2Fpersonal%2Fjb50_aub_edu_lb%2FDocuments%2FRecordings%2FData-Driven+Modeling+Lecture-20260310_140310-Meeting+Recording.mp4%3Fweb%3D1&iCalUid=040000008200E00074C5B7101A82E00800000000FF26CDD361B0DC01000000000000000010000000875130CF0E475545AEE6C30BC46A7FC2&threadId=19%3Ameeting_ODY1ZWJmYWEtMTI0Ny00MTRlLWJkNjItODkzZTcyNzVhYmEz%40thread.v2&organizerId=879bbc5b-4fb7-4d7a-9527-40ff9edea65e&tenantId=c7ba5b1a-41b6-43e9-a120-6ff654ada137&callId=d4b1f5f0-3c42-42dd-a327-f3ad415ce7fb&threadType=meeting&meetingType=Scheduled&subType=RecapSharingLink_RecapCore
hide_from_announcments: false
folder: 14_svd_autoencoders
---
