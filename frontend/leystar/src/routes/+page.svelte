<!-- src/routes/+page.svelte -->
<script lang="ts">
    import Banner from '$lib/components/Banner.svelte';
    import InfoBox from '$lib/components/InfoBox.svelte';
</script>

<!-- Background Layer -->
<div class="background-layer">
    <img src="/background.png" alt="" />
</div>

<!-- Content -->
<div class="content">
    <Banner />
    <div class="spacing">
        <InfoBox />
    </div>
</div>

<style>
    .background-layer {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        overflow: hidden;
    }

    .background-layer img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .content {
        position: relative;
        z-index: 1;
    }

    .spacing {
        margin-top: 2rem; /* Adds spacing between Banner and InfoBox */
    }
</style>